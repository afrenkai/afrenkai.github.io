const std = @import("std");
const serve = @import("serve.zig");

const default_host = "127.0.0.1";
const default_port: u16 = 3000;

const Options = struct {
    host: []const u8 = default_host,
    port: u16 = default_port,
    root: ?[]const u8 = null,
};

pub fn main(init: std.process.Init) !void {
    const options = parseOptions(init) catch |err| switch (err) {
        error.HelpRequested => return,
        else => return err,
    };
    const root = options.root orelse ".";
    var root_dir = try std.Io.Dir.cwd().openDir(init.io, root, .{
        // rust could never
        .follow_symlinks = false,
        .access_sub_paths = true,
    });
    defer root_dir.close(init.io);

    const address = try std.Io.net.IpAddress.parse(options.host, options.port);
    const config: serve.ServerConfig = .{
        .io = init.io,
        .root = root_dir,
    };
    try serve.run(&config, address);
}

fn parseOptions(init: std.process.Init) !Options {
    var options = Options{};
    var args = std.process.Args.iterate(init.minimal.args);
    _ = args.next();

    while (args.next()) |raw_argument| {
        const argument = raw_argument[0..raw_argument.len];
        if (std.mem.eql(u8, argument, "--help")) {
            printUsage();
            return error.HelpRequested;
        }
        if (std.mem.startsWith(u8, argument, "--host=")) {
            options.host = argument[7..];
            continue;
        }
        if (std.mem.startsWith(u8, argument, "--port=")) {
            options.port = try parsePort(argument[7..]);
            continue;
        }
        if (std.mem.startsWith(u8, argument, "--root=")) {
            options.root = argument[7..];
            continue;
        }
        return error.InvalidArgument;
    }

    return options;
}

fn parsePort(text: []const u8) !u16 {
    const port = std.fmt.parseInt(u16, text, 10) catch return error.InvalidPort;
    if (port == 0) return error.InvalidPort;
    return port;
}

fn printUsage() void {
    std.debug.print(
        "usage: site-server [--host=ADDRESS] [--port=PORT] [--root=DIRECTORY]\n" ++
            "defaults: --host=127.0.0.1 --port=8765 --root=.\n",
        .{},
    );
}

test "parses defaults" {
    const options = try parsePort("3000");
    try std.testing.expectEqual(@as(u16, 3000), options);
}

test "rejects invalid ports" {
    try std.testing.expectError(error.InvalidPort, parsePort("0"));
    try std.testing.expectError(error.InvalidPort, parsePort("not-a-port"));
}
