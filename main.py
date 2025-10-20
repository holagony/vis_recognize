import logging
from flask import Flask, jsonify
from flask_api.vis_flask import visibility_module

app = Flask(__name__)
app.register_blueprint(visibility_module, url_prefix='/ai_inference')


# 跨域支持
def after_request(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


app.after_request(after_request)


@app.errorhandler(500)
def bad_request(error):
    response = {'code': 500, 'msg': str(error.original_exception), 'data': {}}
    # return jsonify({"msg": "Bad Request", "status": 400}), 400
    return jsonify(response)


@app.before_request
def process_request():
    # request session redirect render_template
    # print("所有请求之前都会执行这个函数")
    pass


if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
