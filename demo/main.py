import ann
import os
import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("html/main.html")

class EvalHandler(tornado.web.RequestHandler):
    def get(self):
        iv = self.get_argument("iv").split(",")
        iv = map(float, iv)
        print "IN", iv
        ov = ann.eval(iv)
        print "OUT", ov

        win_score = 0
        win_n = None
        for n, score in enumerate(ov):
            if score > win_score:
                win_n = n
                win_score = score

        self.write(str(win_n))

application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/eval", EvalHandler),
],
debug=True,
static_path=os.path.join(os.path.dirname(__file__), "static"))

if __name__ == "__main__":
    application.listen(81)
    tornado.ioloop.IOLoop.instance().start()

