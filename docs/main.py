import time
import tornado.ioloop
import tornado.web
from pymongo import MongoClient
from tornado.web import Application, RequestHandler

class MainHandler(RequestHandler):

    def get(self):
        """Evaluate an input vector using the neural network."""
        #iv = self.get_argument("iv")
        # TODO(jhibberd) Validate
        # TODO(jhibberd) Annotate the output vector?
        self.render("html/main.html")


application = Application([
    (r"/", MainHandler),
], debug=True)

if __name__ == "__main__":
    application.listen(8001)
    tornado.ioloop.IOLoop.instance().start()

