const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    const root = std.Io.Dir.cwd();
    try root.createDirPath(io, "reading-list");

    const result = std.process.run(allocator, io, .{
        .argv = &.{
            "pandoc",
            "content/reading-list.md",
            "--from",
            "markdown",
            "--to",
            "html5",
            "--standalone",
            "--template",
            "templates/reading-list.html",
            "--output",
            "reading-list/index.html",
        },
        .cwd = .{ .dir = root },
        .stderr_limit = .limited(64 * 1024),
        .stdout_limit = .limited(64 * 1024),
    }) catch |err| {
        std.debug.print("error: could not start pandoc for reading list: {s}. Check PATH.\n", .{@errorName(err)});
        std.process.exit(1);
    };
    switch (result.term) {
        .exited => |code| if (code != 0) {
            std.debug.print("pandoc failed for reading list with exit code {d}:\n{s}\n", .{ code, result.stderr });
            std.process.exit(1);
        },
        else => {
            std.debug.print("error: pandoc did not exit normally for reading list.\n", .{});
            std.process.exit(1);
        },
    }
    std.debug.print("Built reading list.\n", .{});
}
