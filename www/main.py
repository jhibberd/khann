import os
import tornado.ioloop
from pymongo import MongoClient
from tornado.web import Application, RequestHandler


class _ClusterSettings(object):
    
    _settings = None

    @classmethod
    def get_settings(cls, nid):
        if not cls._settings:
            cls._settings = {}
            for doc in MongoClient().khann__system.settings.find():
                cls._settings[doc["_id"]] = doc
        return cls._settings[nid]


class XORHandler(RequestHandler):
    def get(self):
        self.render(
            "html/eval/xor.html", 
            settings=_ClusterSettings.get_settings("xor"))


class BinAddHandler(RequestHandler):
    def get(self):
        self.render(
            "html/eval/binadd.html",
            settings=_ClusterSettings.get_settings("binadd"))


class NumHandler(RequestHandler):
    def get(self):
        self.render(
            "html/eval/num.html",
            settings=_ClusterSettings.get_settings("num"))


class AlphaNumHandler(RequestHandler):
    def get(self):
        self.render(
            "html/eval/alphanum.html",
            settings=_ClusterSettings.get_settings("alphanum"))


class MainHandler(RequestHandler):
    """Render demo framework."""

    def get(self):
        self.render("html/main2.html")

'''
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
'''

application = Application([
    (r"/xor",       XORHandler),
    (r"/binadd",    BinAddHandler),
    (r"/num",       NumHandler),
    (r"/alphanum",  AlphaNumHandler),
    (r"/",          MainHandler),
    #(r"/eval", EvalHandler),
],
debug=True,
static_path=os.path.join(os.path.dirname(__file__), "static"))

if __name__ == "__main__":
    application.listen(8002)
    tornado.ioloop.IOLoop.instance().start()

