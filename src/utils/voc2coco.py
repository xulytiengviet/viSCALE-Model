import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re


def get_label2id(labels_path: str) -> Dict[str, int]:
    """Đọc danh sách nhãn và tạo ánh xạ nhãn → ID, bắt đầu từ 1."""
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str) + 1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '',
                 annpaths_list_path: str = None) -> List[str]:
    """Tạo danh sách đường dẫn đến các tệp chú thích Pascal VOC."""
    # Nếu dùng trực tiếp danh sách đường dẫn tệp chú thích.
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r') as f:
            ann_paths = f.read().split()
        return ann_paths

    # Nếu dùng danh sách ID của tệp chú thích.
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    ann_paths = [os.path.join(ann_dir_path, aid + ext_with_dot) for aid in ann_ids]
    return ann_paths


def get_image_info(annotation_root, extract_num_from_imgid=True):
    """Trích xuất thông tin ảnh từ cây XML chú thích Pascal VOC."""
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        numbers = re.findall(r'\d+', img_id)
        if not numbers:
            raise ValueError(
                f"Không thể trích xuất ID dạng số từ tên ảnh: {img_name}"
            )
        img_id = int(numbers[0])

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    """Chuyển một đối tượng Pascal VOC thành một annotation COCO."""
    label = obj.findtext('name')
    assert label in label2id, f"Lỗi: nhãn '{label}' không tồn tại trong label2id!"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin'))) - 1
    ymin = int(float(bndbox.findtext('ymin'))) - 1
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))
    assert xmax > xmin and ymax > ymin, (
        "Lỗi kích thước bounding box! "
        f"(xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    )
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        # Script này chỉ chuyển bounding box, không dùng cho segmentation.
        'segmentation': []
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    """Chuyển tập tệp Pascal VOC XML sang một tệp COCO JSON."""
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    # ID bounding box bắt đầu từ 1.
    bnd_id = 1
    print('Bắt đầu chuyển đổi Pascal VOC XML → COCO JSON...')

    for a_path in tqdm(annotation_paths):
        # Đọc tệp chú thích XML.
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(
            annotation_root=ann_root,
            extract_num_from_imgid=extract_num_from_imgid
        )
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {
            'supercategory': 'none',
            'id': label_id,
            'name': label
        }
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)

    print(f"Hoàn tất. Đã ghi dữ liệu COCO vào: {output_jsonpath}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Chuyển đổi tệp chú thích Pascal VOC dạng XML sang '
            'định dạng COCO JSON.'
        )
    )
    parser.add_argument(
        '--ann_dir',
        type=str,
        default=None,
        help=(
            'Đường dẫn đến thư mục chứa tệp chú thích. '
            'Không cần khi dùng --ann_paths_list.'
        )
    )
    parser.add_argument(
        '--ann_ids',
        type=str,
        default=None,
        help=(
            'Đường dẫn đến tệp chứa danh sách ID chú thích. '
            'Không cần khi dùng --ann_paths_list.'
        )
    )
    parser.add_argument(
        '--ann_paths_list',
        type=str,
        default=None,
        help=(
            'Đường dẫn đến tệp chứa danh sách đường dẫn chú thích. '
            'Không cần khi dùng đồng thời --ann_dir và --ann_ids.'
        )
    )
    parser.add_argument(
        '--labels',
        type=str,
        default=None,
        help='Đường dẫn đến tệp danh sách nhãn.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.json',
        help='Đường dẫn tệp COCO JSON đầu ra.'
    )
    parser.add_argument(
        '--ext',
        type=str,
        default='',
        help='Phần mở rộng bổ sung của tệp chú thích, ví dụ: xml.'
    )
    parser.add_argument(
        '--extract_num_from_imgid',
        action='store_true',
        help='Trích xuất ID dạng số từ tên tệp ảnh.'
    )
    args = parser.parse_args()

    label2id = get_label2id(labels_path=args.labels)
    ann_paths = get_annpaths(
        ann_dir_path=args.ann_dir,
        ann_ids_path=args.ann_ids,
        ext=args.ext,
        annpaths_list_path=args.ann_paths_list
    )
    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=args.output,
        extract_num_from_imgid=args.extract_num_from_imgid
    )


if __name__ == '__main__':
    main()
