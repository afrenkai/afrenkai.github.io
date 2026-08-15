const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const module = b.createModule(.{
        .root_source_file = b.path("zig_backend/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const executable = b.addExecutable(.{
        .name = "site-server",
        .root_module = module,
    });
    b.installArtifact(executable);

    const run_command = b.addRunArtifact(executable);
    if (b.args) |args| run_command.addArgs(args);
    const run_step = b.step("run", "run overengineered zig static site");
    run_step.dependOn(&run_command.step);

    const blog_module = b.createModule(.{
        .root_source_file = b.path("zig_backend/build_blogs.zig"),
        .target = target,
        .optimize = optimize,
    });
    const blog_generator = b.addExecutable(.{
        .name = "blog-maker",
        .root_module = blog_module,
    });
    const blog_run = b.addRunArtifact(blog_generator);
    const blog_step = b.step("blogs", "compile blogs with pandoc");
    blog_step.dependOn(&blog_run.step);

    const main_tests = b.addTest(.{ .root_module = module });
    const blog_tests = b.addTest(.{ .root_module = blog_module });
    const serve_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig_backend/serve.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const test_step = b.step("test", "Run test for server");
    test_step.dependOn(&b.addRunArtifact(main_tests).step);
    test_step.dependOn(&b.addRunArtifact(blog_tests).step);
    test_step.dependOn(&b.addRunArtifact(serve_tests).step);
}
