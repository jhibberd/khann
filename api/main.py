import time
import tornado.ioloop
import tornado.web
from pymongo import MongoClient
from tornado.web import Application, RequestHandler

DATABASE = "khann_alphanum"

"""Geolocate IP?
http://api.hostip.info/get_html.php?ip=118.100.75.63&position=true
"""

class NetworkHandler(RequestHandler):

    def get(self):
        """Evaluate an input vector using the neural network."""
        iv = self.get_argument("iv")
        # TODO(jhibberd) Validate
        # TODO(jhibberd) Annotate the output vector?
        self.write("Hello, world")

    def post(self):
        """Store a new training case, provided by the community."""

        iv = self.get_argument("iv")
        ov = self.get_argument("ov")

        # TODO(jhibberd) Validation

        iv = map(float, iv.split(","))
        ov = map(float, ov.split(","))

        conn = MongoClient()
        db = conn[DATABASE]
        coll = db.training
        doc_id = coll.save({
            "iv":   iv,
            "ov":   ov,
            })

        coll = db.log
        coll.save({
            "_id":          doc_id,
            "created_time": long(time.time()),
            })

        self.write("Hello, world")


application = Application([
    (r"/", NetworkHandler),
], debug=True)

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()

