# Personal-Site-Real

## Local 

```bash
./build.sh serve --root=. --host=127.0.0.1 --port=whatever
```

test:

```bash
./build.sh test
```

## Prod
Run build and server:

```bash
./build.sh release
zig-out/bin/site-server --root=. --host=127.0.0.1 --port=whatever
```
