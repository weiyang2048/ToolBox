#!/usr/bin/env python3
"""ProxyCommand helper: open a TCP tunnel to host:port for ssh/git.

This VM's egress goes through an authenticated HTTP proxy (the
https_proxy env var carries the proxy URL incl. credentials). Direct
TCP to github.com:22 / ssh.github.com:443 is reset by the egress layer,
so git-over-SSH needs an HTTP CONNECT tunnel. Usage (from ssh/git):

    ssh -o ProxyCommand="python3 ssh_proxy_connect.py %h %p" ...

Reads the proxy URL from https_proxy / HTTPS_PROXY / all_proxy /
ALL_PROXY at runtime (credentials are never hardcoded). If no proxy is
configured, connects directly. Exits nonzero on any tunnel failure.
"""

import base64
import os
import select
import socket
import sys
import urllib.parse


def proxy_url():
    for var in ("https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        v = os.environ.get(var)
        if v:
            return v
    return None


def open_tunnel(host, port):
    url = proxy_url()
    if not url:
        return socket.create_connection((host, port), timeout=30)
    p = urllib.parse.urlparse(url)
    phost, pport = p.hostname, p.port or 3128
    s = socket.create_connection((phost, pport), timeout=30)
    lines = ["CONNECT %s:%d HTTP/1.1" % (host, port),
             "Host: %s:%d" % (host, port)]
    if p.username:
        cred = "%s:%s" % (urllib.parse.unquote(p.username),
                          urllib.parse.unquote(p.password or ""))
        lines.append("Proxy-Authorization: Basic " +
                     base64.b64encode(cred.encode()).decode())
    s.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
    # Read the response headers.
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = s.recv(4096)
        if not chunk:
            raise OSError("proxy closed connection during CONNECT")
        buf += chunk
    status = buf.split(b"\r\n", 1)[0].decode("latin-1")
    if " 200 " not in status:
        raise OSError("proxy CONNECT failed: %s" % status)
    # Any bytes past the header terminator belong to the tunnel; in
    # practice the server speaks first only after we send, and recv
    # chunking keeps this empty. Discard defensively is wrong here —
    # push them back by returning them alongside the socket.
    rest = buf.split(b"\r\n\r\n", 1)[1]
    return s, rest


def main():
    host, port = sys.argv[1], int(sys.argv[2])
    res = open_tunnel(host, port)
    if isinstance(res, tuple):
        s, pending = res
    else:
        s, pending = res, b""
    try:
        if pending:
            sys.stdout.buffer.write(pending)
            sys.stdout.buffer.flush()
        fds = [sys.stdin.buffer, s]
        while True:
            r, _, _ = select.select(fds, [], [])
            if sys.stdin.buffer in r:
                data = os.read(0, 65536)
                if not data:
                    s.shutdown(socket.SHUT_WR)
                    fds.remove(sys.stdin.buffer)
                else:
                    s.sendall(data)
            if s in r:
                data = s.recv(65536)
                if not data:
                    break
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            if not fds:
                break
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        try:
            s.close()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
