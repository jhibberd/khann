import ann
import os
import tornado.ioloop
import tornado.web


class MainHandler(tornado.web.RequestHandler):
    """Render demo framework."""

    def get(self):
        self.render("html/main.html")


class EvalHandler(tornado.web.RequestHandler):
    """Evaluate an image expressed as an input vector using the artificial
    neural network.
    """

    def get(self):
        iv = self.get_argument("iv").split(",")
        iv = map(float, iv)
        ov = ann.eval(iv)
        digit = self._pick_strongest(ov)
        self.write(str(digit))

    def _pick_strongest(self, ov):
        """Return the digit indicated by the strongest value in the output
        vector.
        """
        xs = [(score, d) for d, score in enumerate(ov)]
        xs.sort()
        return xs[-1][1]


application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/eval", EvalHandler),
],
debug=True,
static_path=os.path.join(os.path.dirname(__file__), "static"))

if __name__ == "__main__":
    application.listen(81)
    tornado.ioloop.IOLoop.instance().start()

