const std = @import("std");

const Io = std.Io;
const Dir = Io.Dir;
const Allocator = std.mem.Allocator;
const content_directory = "content/blogs";
const output_directory = "blogs";
const work_directory = "blogs/.bloggen-work";
const template_path = "templates/blog-post.html";
const index_file = "blogs/index.html";
const markdown_extension = ".md";
const max_source_bytes = 16 * 1024 * 1024;

const Metadata = struct {
    title: ?[]const u8 = null,
    description: ?[]const u8 = null,
    date: ?[]const u8 = null,
    slug: ?[]const u8 = null,
};

const ParsedDocument = struct {
    metadata: Metadata,
    body: []const u8,
};

const Post = struct {
    title: []const u8,
    description: []const u8,
    date: []const u8,
    slug: []const u8,
};

//I think will always be arena but better to have it be of mem.Allocator.
// not sure I love these syscalls to Io.Dir, but it's fine
const Generator = struct {
    allocator: Allocator,
    io: Io,
    root: Dir,

    pub fn run(self: *const Generator) !void {
        try self.root.createDirPath(self.io, output_directory);
        try self.root.deleteTree(self.io, work_directory);
        try self.root.createDirPath(self.io, work_directory);

        var posts = std.array_list.Managed(Post).init(self.allocator);
        defer posts.deinit();

        var content_dir = self.root.openDir(self.io, content_directory, .{ .iterate = true }) catch |err| switch (err) {
            error.FileNotFound => blk: {
                try self.root.createDirPath(self.io, content_directory);
                break :blk try self.root.openDir(self.io, content_directory, .{ .iterate = true });
            },
            else => return err,
        };
        defer content_dir.close(self.io);

        var entries = std.array_list.Managed([]const u8).init(self.allocator);
        defer entries.deinit();
        var iterator = content_dir.iterate();
        while (try iterator.next(self.io)) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, markdown_extension)) {
                try entries.append(try self.allocator.dupe(u8, entry.name));
            }
        }
        std.mem.sort([]const u8, entries.items, {}, lessThanBytes);

        for (entries.items) |entry_name| {
            const source_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ content_directory, entry_name });

            const source = try self.root.readFileAlloc(self.io, source_path, self.allocator, .limited(max_source_bytes));

            const parsed = parseDocument(source);

            if (std.mem.trim(u8, parsed.body, " \t\r\n").len == 0) {
                continue;
            }

            const basename = entry_name[0 .. entry_name.len - markdown_extension.len];
            const inferred_title = firstHeading(parsed.body) orelse basename;
            const title = if (parsed.metadata.title) |value| if (value.len != 0) value else inferred_title else inferred_title;
            const base_slug = try slugify(self.allocator, parsed.metadata.slug orelse title);
            const slug = try uniqueSlug(self.allocator, posts.items, base_slug);
            const description = if (parsed.metadata.description) |value| value else firstParagraph(self.allocator, parsed.body);
            const date = parsed.metadata.date orelse "";
            const body = if (parsed.metadata.title != null) parsed.body else stripFirstHeading(parsed.body);

            try self.buildPost(title, description, date, slug, body);
            try posts.append(.{ .title = title, .description = description, .date = date, .slug = slug });
        }

        std.mem.sort(Post, posts.items, {}, postLessThan);
        try self.removeStalePosts(posts.items);
        const index = try makeIndex(self.allocator, posts.items);
        try writeFileAtomic(self.root, self.io, index_file, index);
        try self.root.deleteTree(self.io, work_directory);

        std.debug.print("Built {d} blog post{s}.\n", .{ posts.items.len, if (posts.items.len == 1) "" else "s" });
    }

    fn buildPost(
        self: *const Generator,
        title: []const u8,
        description: []const u8,
        date: []const u8,
        slug: []const u8,
        body: []const u8,
    ) !void {
        const normalized_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}.md", .{ work_directory, slug });
        try writeFileAtomic(self.root, self.io, normalized_path, body);

        const post_directory = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ output_directory, slug });
        try self.root.deleteTree(self.io, post_directory);
        try self.root.createDirPath(self.io, post_directory);
        const output_path = try std.fmt.allocPrint(self.allocator, "{s}/index.html", .{post_directory});
        const title_metadata = try std.fmt.allocPrint(self.allocator, "title={s}", .{title});
        const description_metadata = if (description.len != 0)
            try std.fmt.allocPrint(self.allocator, "description={s}", .{description})
        else
            null;
        const date_metadata = if (date.len != 0)
            try std.fmt.allocPrint(self.allocator, "date={s}", .{date})
        else
            null;
        var arguments: [20][]const u8 = undefined;
        var argument_count: usize = 0;
        arguments[argument_count] = "pandoc";
        argument_count += 1;
        arguments[argument_count] = normalized_path;
        argument_count += 1;
        arguments[argument_count] = "--from";
        argument_count += 1;
        arguments[argument_count] = "markdown+tex_math_dollars+tex_math_single_backslash+raw_tex";
        argument_count += 1;
        arguments[argument_count] = "--to";
        argument_count += 1;
        arguments[argument_count] = "html5";
        argument_count += 1;
        arguments[argument_count] = "--standalone";
        argument_count += 1;
        arguments[argument_count] = "--mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js";
        argument_count += 1;
        arguments[argument_count] = "--template";
        argument_count += 1;
        arguments[argument_count] = template_path;
        argument_count += 1;
        arguments[argument_count] = "--output";
        argument_count += 1;
        arguments[argument_count] = output_path;
        argument_count += 1;
        arguments[argument_count] = "--metadata";
        argument_count += 1;
        arguments[argument_count] = title_metadata;
        argument_count += 1;
        if (description_metadata) |metadata| {
            arguments[argument_count] = "--metadata";
            argument_count += 1;
            arguments[argument_count] = metadata;
            argument_count += 1;
        }
        if (date_metadata) |metadata| {
            arguments[argument_count] = "--metadata";
            argument_count += 1;
            arguments[argument_count] = metadata;
            argument_count += 1;
        }
        const result = try std.process.run(self.allocator, self.io, .{
            .argv = arguments[0..argument_count],
            .cwd = .{ .dir = self.root },
            .stderr_limit = .limited(64 * 1024),
            .stdout_limit = .limited(64 * 1024),
        });
        switch (result.term) {
            .exited => |code| if (code != 0) {
                std.debug.print("pandoc failed for {s} with exit code {d}:\n{s}\n", .{ slug, code, result.stderr });
                return error.PandocFailed;
            },
            else => {
                std.debug.print("pandoc did not exit normally for {s}\n", .{slug});
                return error.PandocFailed;
            },
        }
    }

    // might come in handy later. Lowkey just wanted to see if ptr to generator worked
    fn removeStalePosts(self: *const Generator, posts: []const Post) !void {
        var output_dir = try self.root.openDir(self.io, output_directory, .{ .iterate = true });
        defer output_dir.close(self.io);
        var iterator = output_dir.iterate();
        while (try iterator.next(self.io)) |entry| {
            if (entry.kind != .directory or std.mem.eql(u8, entry.name, ".bloggen-work")) continue;
            if (containsSlug(posts, entry.name)) continue;
            const stale_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ output_directory, entry.name });
            try self.root.deleteTree(self.io, stale_path);
        }
    }
};

pub fn main(init: std.process.Init) !void {
    const generator: Generator = .{
        .allocator = init.arena.allocator(),
        .io = init.io,
        .root = Dir.cwd(),
    };
    try generator.run();
}

fn writeFileAtomic(dir: Dir, io: Io, path: []const u8, data: []const u8) !void {
    var file = try dir.createFileAtomic(io, path, .{ .make_path = true, .replace = true });
    defer file.deinit(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.file.writer(io, &buffer);
    try writer.interface.writeAll(data);
    try writer.flush();
    try file.replace(io);
}

//stole from old node code
fn parseDocument(source: []const u8) ParsedDocument {
    if (!std.mem.startsWith(u8, source, "---\n")) return .{ .metadata = .{}, .body = source };
    const end = std.mem.findPos(u8, source, 4, "\n---") orelse return .{ .metadata = .{}, .body = source };
    const raw = std.mem.trim(u8, source[4..end], " \t\r\n");
    var metadata: Metadata = .{};
    var lines = std.mem.splitScalar(u8, raw, '\n');
    while (lines.next()) |line| {
        const separator = std.mem.findScalar(u8, line, ':') orelse continue;
        const key = std.mem.trim(u8, line[0..separator], " \t\r");
        const value = trimQuoted(std.mem.trim(u8, line[separator + 1 ..], " \t\r"));
        if (!validMetadataKey(key)) continue;
        if (std.mem.eql(u8, key, "title")) metadata.title = value;
        if (std.mem.eql(u8, key, "description")) metadata.description = value;
        if (std.mem.eql(u8, key, "date")) metadata.date = value;
        if (std.mem.eql(u8, key, "slug")) metadata.slug = value;
    }
    var body = source[end + 5 ..];
    if (body.len != 0 and body[0] == '\n') body = body[1..];
    return .{ .metadata = metadata, .body = body };
}

fn validMetadataKey(key: []const u8) bool {
    if (key.len == 0) return false;
    for (key) |byte| {
        if (!std.ascii.isAlphanumeric(byte) and byte != '_' and byte != '-') return false;
    }
    return true;
}

fn trimQuoted(value: []const u8) []const u8 {
    if (value.len >= 2 and ((value[0] == '\'' and value[value.len - 1] == '\'') or
        (value[0] == '"' and value[value.len - 1] == '"'))) return value[1 .. value.len - 1];
    return value;
}

fn firstHeading(body: []const u8) ?[]const u8 {
    var lines = std.mem.splitScalar(u8, body, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "# ")) return std.mem.trim(u8, line[2..], " \t\r");
    }
    return null;
}

fn firstParagraph(allocator: Allocator, body: []const u8) []const u8 {
    var blocks = std.mem.splitSequence(u8, body, "\n\n");
    while (blocks.next()) |block| {
        const paragraph = std.mem.trim(u8, block, " \t\r\n");
        if (paragraph.len == 0 or std.mem.startsWith(u8, paragraph, "# ") or
            std.mem.startsWith(u8, paragraph, "$$") or std.mem.startsWith(u8, paragraph, "-") or
            std.mem.startsWith(u8, paragraph, "!")) continue;
        var description: [180]u8 = undefined;
        var length: usize = 0;
        var whitespace = false;
        for (paragraph) |byte| {
            if (std.ascii.isWhitespace(byte)) {
                whitespace = true;
                continue;
            }
            if (whitespace and length != 0) {
                if (length == description.len) break;
                description[length] = ' ';
                length += 1;
            }
            if (length == description.len) break;
            description[length] = byte;
            length += 1;
            whitespace = false;
        }
        return allocator.dupe(u8, description[0..length]) catch return "";
    }
    return "";
}

fn stripFirstHeading(body: []const u8) []const u8 {
    if (!std.mem.startsWith(u8, body, "# ")) return body;
    const line_end = std.mem.findScalar(u8, body, '\n') orelse return "";
    return std.mem.trim(u8, body[line_end + 1 ..], "\n");
}

fn slugify(allocator: Allocator, value: []const u8) ![]const u8 {
    var writer = std.Io.Writer.Allocating.init(allocator);
    var separator = false;
    for (value) |byte| {
        if (std.ascii.isAlphanumeric(byte)) {
            if (separator) try writer.writer.writeByte('-');
            try writer.writer.writeByte(std.ascii.toLower(byte));
            separator = false;
        } else if (writer.written().len != 0) {
            separator = true;
        }
    }
    var result = writer.written();
    while (result.len != 0 and result[result.len - 1] == '-') result = result[0 .. result.len - 1];
    if (result.len == 0) return allocator.dupe(u8, "post");
    return result;
}

fn uniqueSlug(allocator: Allocator, posts: []const Post, base: []const u8) ![]const u8 {
    if (!containsSlug(posts, base)) return base;
    var suffix: usize = 2;
    while (true) : (suffix += 1) {
        const candidate = try std.fmt.allocPrint(allocator, "{s}-{d}", .{ base, suffix });
        if (!containsSlug(posts, candidate)) return candidate;
    }
}

fn makeIndex(allocator: Allocator, posts: []const Post) ![]const u8 {
    var writer = std.Io.Writer.Allocating.init(allocator);
    try writer.writer.writeAll(
        "<!doctype html>\n<html lang=\"en\">\n  <head>\n" ++
            "    <meta charset=\"utf-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" ++
            "    <meta name=\"robots\" content=\"noindex, nofollow, noarchive, nosnippet, noimageindex, notranslate\">\n" ++
            "    <meta name=\"author\" content=\"Artem Frenk\">\n    <meta name=\"description\" content=\"Blog posts and notes by Artem Frenk.\">\n" ++
            "    <title>Blogs | Artem Frenk</title>\n    <link rel=\"stylesheet\" href=\"../assets/site.css\">\n" ++
            "  </head>\n  <body>\n    <div class=\"site-shell\">\n      <header class=\"masthead\">\n" ++
            "        <nav class=\"site-nav\" aria-label=\"Primary navigation\">\n          <a href=\"/\">Home</a>\n" ++
            "          <a href=\"/papers/\">Papers</a>\n          <a href=\"/blogs/\" aria-current=\"page\">Blogs and Notes</a>\n" ++
            "        </nav>\n      </header>\n\n      <main id=\"content\" class=\"page-grid page-grid-single\">\n        <article>\n" ++
            "          <p class=\"kicker\">Blogs</p>\n          <h1 class=\"page-title\">Blogs</h1>\n\n" ++
            "          <ul class=\"item-list\">\n",
    );
    if (posts.len == 0) {
        try writer.writer.writeAll(
            "            <li>\n              <h3>No posts yet</h3>\n" ++
                "              <p>Markdown posts added to <code>content/blogs/</code> will appear here after running <code>zig build blogs</code>.</p>\n" ++
                "            </li>\n",
        );
    } else {
        for (posts) |post| {
            try writer.writer.writeAll("            <li>\n              <h3><a href=\"");
            try writeHtmlEscaped(&writer.writer, post.slug);
            try writer.writer.writeAll("/\">");
            try writeHtmlEscaped(&writer.writer, post.title);
            try writer.writer.writeAll("</a></h3>\n");
            if (post.date.len != 0) {
                try writer.writer.writeAll("              <p class=\"meta\">");
                try writeHtmlEscaped(&writer.writer, post.date);
                try writer.writer.writeAll("</p>\n");
            }
            try writer.writer.writeAll("            </li>\n");
        }
    }
    try writer.writer.writeAll(
        "          </ul>\n        </article>\n      </main>\n    </div>\n  </body>\n</html>\n",
    );
    return writer.written();
}

fn writeHtmlEscaped(writer: *Io.Writer, value: []const u8) !void {
    for (value) |byte| switch (byte) {
        '&' => try writer.writeAll("&amp;"),
        '<' => try writer.writeAll("&lt;"),
        '>' => try writer.writeAll("&gt;"),
        '"' => try writer.writeAll("&quot;"),
        else => try writer.writeByte(byte),
    };
}

fn containsSlug(posts: []const Post, slug: []const u8) bool {
    for (posts) |post| if (std.mem.eql(u8, post.slug, slug)) return true;
    return false;
}

fn lessThanBytes(_: void, left: []const u8, right: []const u8) bool {
    return std.mem.order(u8, left, right) == .lt;
}

fn postLessThan(_: void, left: Post, right: Post) bool {
    if (left.date.len != 0 and right.date.len != 0 and !std.mem.eql(u8, left.date, right.date)) {
        return std.mem.order(u8, right.date, left.date) == .lt;
    }
    if (left.date.len != 0 and right.date.len == 0) return true;
    if (left.date.len == 0 and right.date.len != 0) return false;
    return std.mem.order(u8, left.title, right.title) == .lt;
}

test "parses front matter and body" {
    const parsed = parseDocument("---\ntitle: Example\ndate: 2026-01-01\n---\n\n# Heading\n\nBody");
    try std.testing.expectEqualStrings("Example", parsed.metadata.title.?);
    try std.testing.expectEqualStrings("2026-01-01", parsed.metadata.date.?);
    try std.testing.expectEqualStrings("# Heading\n\nBody", parsed.body);
}

test "slugifies and disambiguates" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const slug = try slugify(allocator, "Hello, Zig!");
    try std.testing.expectEqualStrings("hello-zig", slug);
    const posts = [_]Post{.{ .title = "", .description = "", .date = "", .slug = "hello-zig" }};
    const second = try uniqueSlug(allocator, &posts, slug);
    try std.testing.expectEqualStrings("hello-zig-2", second);
}
