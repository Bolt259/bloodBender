from http.server import BaseHTTPRequestHandler, HTTPServer


class OurHTTPServer(HTTPServer):
    path: str


class HTTPServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(
            bytes("<html><h1>You may now close this window.</h1></html>", "utf-8")
        )
        self.server.path = self.path

    # Disable logging from the HTTP Server
    def log_message(self, format, *args):
        return


class RedirectCatcher:
    FORMAT = "http://localhost:{port}/callback"

    def __init__(self, port: int) -> None:
        self.server = OurHTTPServer(("localhost", port), HTTPServerHandler)
        self.redirect_uri = self.FORMAT.format(port=port)

    def catch(self) -> str:
        with self.server:
            self.server.handle_request()
        return self.server.path
