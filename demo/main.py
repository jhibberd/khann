import khann
import os
import sys
import tornado.ioloop
from pymongo import MongoClient
from tornado.web import Application, RequestHandler


class _ClusterMeta(object):
    """Access to cluster settings/metadata for use in input validation."""

    _meta = None

    @classmethod
    def is_valid_NID(cls, nid):
        return nid in cls._get_meta()

    @classmethod
    def get_topology(cls, nid):
        return cls._get_meta()[nid]["topology"]

    @classmethod
    def _get_meta(cls):
        if not cls._meta:
            cls._meta = {}
            for doc in MongoClient().khann__system.settings.find():
                cls._meta[doc["_id"]] = doc
        return cls._meta


class _EvalHandler(RequestHandler):

    def get(self, nid):
        """Evaluate an input vector using a trained neural network in the
        in-memory cluster.
        """
        iv = self.get_argument("iv")
        self._validate_NID(nid)
        iv = self._fmt_and_validate_IV(nid, iv)
        print nid, iv
        ov = khann.cluster_eval(nid, iv)
        print "test"
        self.write({
            "ov_real":  ov,
            "ov_bin":   self._bin_OV(ov),
            })

    @staticmethod
    def _bin_OV(ov):
        return map(lambda x: 1 if x >= 0.5 else 0, ov)

    @staticmethod
    def _validate_NID(nid):
        if not _ClusterMeta.is_valid_NID(nid):
            raise Exception("Unknown network '%s'" % nid)

    @staticmethod
    def _fmt_and_validate_IV(nid, iv):
        iv = iv.split(",")
        try:
            iv = map(float, iv)
        except ValueError:
            raise Exception("Input vector contains non-float values")
        expected_len = _ClusterMeta.get_topology(nid)[0]
        if len(iv) != expected_len:
            raise Exception("Input vector length should be %d but is %d" % \
                (expected_len, len(iv)))
        return iv


class _MainHandler(RequestHandler):
    def get(self):
        self.render("main.html")


if __name__ == "__main__":
    khann.cluster_init()
    try:
        application = Application([
            (r"/",              _MainHandler),
            (r"/eval/(\w*)/?",  _EvalHandler),
            ],
            debug=True, 
            static_path=os.path.join(os.path.dirname(__file__), "static")) 
        application.listen(port=int(sys.argv[1]))
        tornado.ioloop.IOLoop.instance().start()
    finally:
        khann.cluster_destroy()

