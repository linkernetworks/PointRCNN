import json
import os
def dump_json(config,out_result):
    pcd_paths = []
    with open(config['data_reader']['input_dir'], 'r') as f:
        data = json.loads(f.read())
        for i,sample in enumerate(data):
            label_list = []
            for key,value in out_result.items():
                for box in value[i]:
                    label_object = {}
                    label_object['category'] = str(key)
                    label_object['attributes'] = []
                    label_object['box3d'] = {
                            "x": float(box[0]),
                            "y": float(box[1]),
                            "z": float(box[2]),
                            "l": float(box[5]),
                            "w": float(box[4]),
                            "h": float(box[3]),
                            "yaw": float(box[6])
                        }
                    label_list.append(label_object)
            sample['labels'] = label_list
        
    import ipdb; ipdb.set_trace()
    if not os.path.exists(config['output']['directory']):
        os.makedirs(config['output']['directory'])
    with open(os.path.join(config['output']['directory'], '%s' % config['output']['file_name']),
              'w') as out_file:
        json.dump(data, out_file)