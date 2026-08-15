const std = @import("std");

const Io = std.Io;
const File = Io.File;
const Dir = Io.Dir;
const HttpServer = std.http.Server;
const HttpStatus = std.http.Status;
const request_buffer_size = 16 * 1024;
const response_buffer_size = 16 * 1024;
const file_buffer_size = 16 * 1024;
const max_connections = 64;
const index_file = "index.html";

pub const ServerConfig = struct {
    io: Io,
    root: Dir,
};

pub fn run(config: *const ServerConfig, address: Io.net.IpAddress) !void {
    var listener = try address.listen(config.io, .{
        .kernel_backlog = 128,
        .reuse_address = true,
    });
    defer listener.deinit(config.io);

    std.debug.print("serving content on port {d}\n", .{address.getPort()});

    var connections: Io.Group = .init;
    defer connections.cancel(config.io);
    var connection_limit: Io.Semaphore = .{ .permits = max_connections };
    //shoutout ben

    while (true) {
        connection_limit.wait(config.io) catch |err| switch (err) {
            error.Canceled => return err,
        };

        const stream = listener.accept(config.io) catch |err| switch (err) {
            error.Canceled => {
                connection_limit.post(config.io);
                return err;
            },
            else => {
                std.debug.print("accept failed: {t}\n", .{err});
                connection_limit.post(config.io);
                continue;
            },
        };

        connections.concurrent(config.io, serveConnection, .{ config, stream, &connection_limit }) catch |err| {
            std.debug.print("connection scheduling failed: {t}\n", .{err});
            stream.close(config.io);
            connection_limit.post(config.io);
        };
    }
}

fn serveConnection(
    config: *const ServerConfig,
    stream: Io.net.Stream,
    connection_limit: *Io.Semaphore,
) void {
    defer connection_limit.post(config.io);
    defer stream.close(config.io);

    //no clue what sane defaults for these are so I inherit
    var request_storage: [request_buffer_size]u8 = undefined;
    var response_storage: [response_buffer_size]u8 = undefined;
    var input = stream.reader(config.io, &request_storage);
    var output = stream.writer(config.io, &response_storage);
    var server: HttpServer = .init(&input.interface, &output.interface);

    var request = server.receiveHead() catch |err| switch (err) {
        error.HttpConnectionClosing => return,
        else => {
            std.debug.print("request rejected: {t}\n", .{err});
            return;
        },
    };

    handleRequest(config, &request) catch |err| {
        std.debug.print("request failed: {t}\n", .{err});
    };
}

fn handleRequest(config: *const ServerConfig, request: *HttpServer.Request) !void {
    switch (request.head.method) {
        .GET, .HEAD => {},
        else => {
            try respondText(request, .method_not_allowed, "method not allowed\n", &.{
                .{ .name = "allow", .value = "GET, HEAD" },
            });
            return;
        },
    }

    // ew but not sure how to handle otherwise
    var relative_path: [Dir.max_path_bytes]u8 = undefined;
    const path = normalizeRequestPath(request.head.target, &relative_path) catch |err| switch (err) {
        error.InvalidRequestPath, error.PathTraversal => {
            try respondText(request, .bad_request, "bad request\n", &.{});
            return;
        },
        error.PathTooLong => {
            try respondText(request, .uri_too_long, "uri too long\n", &.{});
            return;
        },
    };

    var file_path: [Dir.max_path_bytes]u8 = undefined;
    var file = openFile(config, path, &file_path) catch |err| switch (err) {
        error.FileNotFound, error.NotDir, error.SymLinkLoop => {
            try respondText(request, .not_found, "not found\n", &.{});
            return;
        },
        error.PathTooLong => {
            try respondText(request, .uri_too_long, "uri too long\n", &.{});
            return;
        },
        error.AccessDenied, error.PermissionDenied => {
            try respondText(request, .forbidden, "forbidden\n", &.{});
            return;
        },
        else => return err,
    };
    defer file.close(config.io);

    const stat = try file.stat(config.io);
    if (stat.kind != .file) {
        try respondText(request, .not_found, "not found\n", &.{});
        return;
    }

    var response_storage: [response_buffer_size]u8 = undefined;
    var response = try request.respondStreaming(&response_storage, .{
        .content_length = stat.size,
        .respond_options = .{
            .keep_alive = false,
            .extra_headers = &.{
                .{ .name = "content-type", .value = contentType(path) },
                .{ .name = "x-content-type-options", .value = "nosniff" },
                .{ .name = "x-robots-tag", .value = "noindex, nofollow, noarchive, nosnippet, noimageindex, notranslate" },
                .{ .name = "referrer-policy", .value = "no-referrer" },
            },
        },
    });

    if (request.head.method != .HEAD) {
        var file_storage: [file_buffer_size]u8 = undefined;
        var file_reader = file.reader(config.io, &file_storage);
        const sent = try response.writer.sendFileAll(&file_reader, .limited(stat.size));
        if (sent != stat.size) return error.FileChanged;
        try response.writer.defaultFlush();
    }
    try response.http_protocol_output.flush();
}

fn respondText(
    request: *HttpServer.Request,
    status: HttpStatus,
    body: []const u8,
    extra_headers: []const std.http.Header,
) !void {
    var headers: [4]std.http.Header = undefined;
    headers[0] = .{ .name = "content-type", .value = "text/plain; charset=utf-8" };
    headers[1] = .{ .name = "x-content-type-options", .value = "nosniff" };
    headers[2] = .{ .name = "x-robots-tag", .value = "noindex, nofollow, noarchive, nosnippet, noimageindex, notranslate" };
    headers[3] = .{ .name = "referrer-policy", .value = "no-referrer" };

    var all_headers: [6]std.http.Header = undefined;
    var header_count: usize = 0;
    for (headers) |header| {
        all_headers[header_count] = header;
        header_count += 1;
    }
    for (extra_headers) |header| {
        all_headers[header_count] = header;
        header_count += 1;
    }

    try request.respond(body, .{
        .status = status,
        .keep_alive = false,
        .extra_headers = all_headers[0..header_count],
    });
}

fn openFile(config: *const ServerConfig, relative_path: []const u8, path_storage: *[Dir.max_path_bytes]u8) !File {
    return config.root.openFile(config.io, relative_path, .{
        .allow_directory = false,
        .follow_symlinks = false,
        .resolve_beneath = true,
    }) catch |err| switch (err) {
        error.IsDir => {
            const with_index = try appendIndex(relative_path, path_storage);
            return config.root.openFile(config.io, with_index, .{
                .allow_directory = false,
                .follow_symlinks = false,
                .resolve_beneath = true,
            });
        },
        else => return err,
    };
}

fn appendIndex(path: []const u8, storage: *[Dir.max_path_bytes]u8) ![]const u8 {
    if (path.len + 1 + index_file.len > storage.len) return error.PathTooLong;

    // DANGER ZONE JUST DROPPED!!!!!
    @memcpy(storage[0..path.len], path);
    var length = path.len;
    if (length == 0 or storage[length - 1] != '/') {
        storage[length] = '/';
        length += 1;
    }
    @memcpy(storage[length .. length + index_file.len], index_file);
    return storage[0 .. length + index_file.len];
}

const NormalizeError = error{
    InvalidRequestPath,
    PathTraversal,
    PathTooLong,
};

fn normalizeRequestPath(target: []const u8, output: *[Dir.max_path_bytes]u8) NormalizeError![]const u8 {
    const query_start = std.mem.findScalar(u8, target, '?') orelse target.len;
    const encoded_path = target[0..query_start];
    if (encoded_path.len == 0 or encoded_path[0] != '/') return error.InvalidRequestPath;
    if (encoded_path.len > output.len) return error.PathTooLong;

    var decoded: [Dir.max_path_bytes]u8 = undefined;
    var decoded_length: usize = 0;
    var encoded_index: usize = 0;
    while (encoded_index < encoded_path.len) {
        const byte = encoded_path[encoded_index];
        if (byte == '%') {
            if (encoded_index + 2 >= encoded_path.len) return error.InvalidRequestPath;
            const high = hexValue(encoded_path[encoded_index + 1]) orelse return error.InvalidRequestPath;
            const low = hexValue(encoded_path[encoded_index + 2]) orelse return error.InvalidRequestPath;
            decoded[decoded_length] = (high << 4) | low;
            decoded_length += 1;
            encoded_index += 3;
        } else {
            decoded[decoded_length] = byte;
            decoded_length += 1;
            encoded_index += 1;
        }
    }

    var output_length: usize = 0;
    var component_start: usize = 1;
    while (component_start <= decoded_length) {
        const separator = std.mem.findScalarPos(u8, decoded[0..decoded_length], component_start, '/') orelse decoded_length;
        const component = decoded[component_start..separator];
        if (component.len != 0) {
            if (std.mem.eql(u8, component, ".")) return error.InvalidRequestPath;
            if (std.mem.eql(u8, component, "..")) return error.PathTraversal;
            for (component) |byte| {
                if (byte == 0 or byte == '\\' or byte < 0x20 or byte == 0x7f) return error.InvalidRequestPath;
            }
            if (output_length != 0) {
                if (output_length + 1 >= output.len) return error.PathTooLong;
                output[output_length] = '/';
                output_length += 1;
            }
            if (output_length + component.len > output.len) return error.PathTooLong;
            @memcpy(output[output_length .. output_length + component.len], component);
            output_length += component.len;
        }
        if (separator == decoded_length) break;
        component_start = separator + 1;
    }

    if (output_length == 0) {
        @memcpy(output[0..index_file.len], index_file);
        return output[0..index_file.len];
    }

    if (decoded_length > 1 and decoded[decoded_length - 1] == '/') {
        if (output_length + 1 + index_file.len > output.len) return error.PathTooLong;
        output[output_length] = '/';
        @memcpy(output[output_length + 1 .. output_length + 1 + index_file.len], index_file);
        output_length += 1 + index_file.len;
    }

    return output[0..output_length];
}

fn hexValue(byte: u8) ?u8 {
    return switch (byte) {
        '0'...'9' => byte - '0',
        'a'...'f' => byte - 'a' + 10,
        'A'...'F' => byte - 'A' + 10,
        else => null,
    };
}

fn contentType(path: []const u8) []const u8 {
    const extension = Dir.path.extension(path);
    if (std.ascii.eqlIgnoreCase(extension, ".css")) return "text/css; charset=utf-8";
    if (std.ascii.eqlIgnoreCase(extension, ".html")) return "text/html; charset=utf-8";
    if (std.ascii.eqlIgnoreCase(extension, ".js")) return "text/javascript; charset=utf-8";
    if (std.ascii.eqlIgnoreCase(extension, ".json")) return "application/json; charset=utf-8";
    if (std.ascii.eqlIgnoreCase(extension, ".md")) return "text/markdown; charset=utf-8";
    if (std.ascii.eqlIgnoreCase(extension, ".pdf")) return "application/pdf";
    if (std.ascii.eqlIgnoreCase(extension, ".png")) return "image/png";
    if (std.ascii.eqlIgnoreCase(extension, ".jpg")) return "image/jpeg";
    if (std.ascii.eqlIgnoreCase(extension, ".jpeg")) return "image/jpeg";
    if (std.ascii.eqlIgnoreCase(extension, ".svg")) return "image/svg+xml; charset=utf-8";
    if (std.ascii.eqlIgnoreCase(extension, ".txt")) return "text/plain; charset=utf-8";
    return "application/octet-stream";
}

test "normalizes safe paths and query strings" {
    var output: [Dir.max_path_bytes]u8 = undefined;
    try std.testing.expectEqualStrings("index.html", try normalizeRequestPath("/?cache=1", &output));
    try std.testing.expectEqualStrings("blogs/index.html", try normalizeRequestPath("/blogs/", &output));
    try std.testing.expectEqualStrings("assets/site.css", try normalizeRequestPath("/assets//site.css?x", &output));
}

test "rejects traversal and malformed escapes" {
    var output: [Dir.max_path_bytes]u8 = undefined;
    try std.testing.expectError(error.PathTraversal, normalizeRequestPath("/%2e%2e/README.md", &output));
    try std.testing.expectError(error.InvalidRequestPath, normalizeRequestPath("/%", &output));
    try std.testing.expectError(error.InvalidRequestPath, normalizeRequestPath("/a%2gb", &output));
}

test "maps content types" {
    try std.testing.expectEqualStrings("text/html; charset=utf-8", contentType("index.html"));
    try std.testing.expectEqualStrings("application/octet-stream", contentType("asset.bin"));
}
