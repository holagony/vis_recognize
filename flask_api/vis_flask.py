import json
import simplejson
from flask import Blueprint, request, jsonify
from flask_api.vis_handler import visibility_inference

visibility_module = Blueprint('visibility_module', __name__)

@visibility_module.route('/v1/visibility', methods=['POST'])
def run_visibility():
    '''
    能见度等级分类
    同步/异步
    '''
    json_str = request.get_data(as_text=True)  # 获取JSON字符串
    data_json = json.loads(json_str)
    result_dict = visibility_inference(data_json)
    return_data = simplejson.dumps({'code': 200, 
                                    'msg': 'success',
                                    'data': result_dict}, ensure_ascii=False, ignore_nan=True)
    return return_data