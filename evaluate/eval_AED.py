from __future__ import division
import os
import re
import shapely
from shapely.geometry import Polygon
import numpy as np

import editdistance
from hanziconv import HanziConv
from tqdm import tqdm


IOU_THRESH = 0.5
N_TEST = 4229


def polygon_from_str(poly_str):
    """
  Create a shapely polygon object from gt or dt line.
  """

    polygon_points = [float(o) for o in poly_str.split(',')[:8]]
    polygon_points = np.array(polygon_points).reshape(4, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon

def extract_num(poly_str):
    res=poly_str.split('\t')
    res1=res[0]
    res2=res[-1]
    return res1,res2

def gt_from_line(gt_line):
    """
  Parse a line of groundtruth.
  """
    # remove possible utf-8 BOM
    if gt_line.startswith('\xef\xbb\xbf'):
        gt_line = gt_line[3:]

    gt_line = gt_line.encode('utf-8')
    gt_line = gt_line.decode('utf-8')
    #print(type(gt_line))
    gt_polygon1 , gt_text = extract_num(gt_line)
    #print(gt_polygon1)
    #print(gt_text)


    gt_polygon = polygon_from_str(gt_polygon1)
    gt = {'polygon': gt_polygon,  'text': gt_text.strip()}
    #print(gt)
    return gt


def dt_from_line(dt_line):
    """
  Parse a line of detection result.
  """
    # remove possible utf-8 BOM
    if dt_line.startswith('\xef\xbb\xbf'):
        dt_line = dt_line[3:]
    dt_line = dt_line.encode('utf-8')
    dt_line = dt_line.decode('utf-8')

    dt_polygon1,dt_text = extract_num(dt_line)

    dt_polygon = polygon_from_str(dt_polygon1)

    dt = {'polygon': dt_polygon, 'text': dt_text}
    #print(dt)
    return dt


def polygon_iou(poly1, poly2):
    """
  Intersection over union between two shapely polygons.
  """
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def normalize_txt(st):
    """
  Normalize Chinese text strings by:
    - remove puncutations and other symbols
    - convert traditional Chinese to simplified
    - convert English chraacters to lower cases
  """
    st = ''.join(st.split(' '))
    st = re.sub("\"", "", st)
    # remove any this not one of Chinese character, ascii 0-9, and ascii a-z and A-Z
    new_st = re.sub('[^\u4e00-\u9fa5\u0041-\u005a\u0061-\u007a0-9]+', '', st)
    # convert Traditional Chinese to Simplified Chinese
    new_st = HanziConv.toSimplified(new_st)
    # convert uppercase English letters to lowercase
    new_st = new_st.lower()
    return new_st


def text_distance(str1, str2):
    str1 = normalize_txt(str1)
    str2 = normalize_txt(str2)
    return editdistance.eval(str1, str2)


def recog_eval(gt_dir, recog_dir):
    '''
    最终计算结果在本函数的返回值中
    '''
    test_gt_files = os.listdir(gt_dir)
    num_test = len(test_gt_files)
    gt_count = 0
    total_dist = 0

    for gt_file in tqdm(test_gt_files):
        # distance calculated on this example
        example_dist = 0

        # load groundtruth
        with open(os.path.join(gt_dir, gt_file),encoding='utf-8-sig') as f:
            gt_lines = f.readlines()
        gts = [gt_from_line(o) for o in gt_lines]
        n_gt = len(gts)

        # load results
        dt_result_file = os.path.join(recog_dir,  gt_file)
        if not os.path.exists(dt_result_file):
            print('{} not found'.format(dt_result_file))
            dts = []
        else:
            with open(dt_result_file,encoding='utf-8-sig') as f:
                dt_lines = f.readlines()
            dts = [dt_from_line(o) for o in dt_lines]
        n_dt = len(dts)

        # match dt index of every gt
        gt_match = np.empty(n_gt, dtype=np.int32)
        gt_match.fill(-1)
        # match gt index of every dt
        dt_match = np.empty(n_dt, dtype=np.int32)
        dt_match.fill(-1)

        # find match for every GT
        for i, gt in enumerate(gts):
            max_iou = 0
            match_dt_idx = -1
            for j, dt in enumerate(dts):
                if dt_match[j] >= 0:
                    # already matched to some GT
                    continue
                iou = polygon_iou(gt['polygon'], dt['polygon'])
                if iou > IOU_THRESH and iou > max_iou:
                    max_iou = iou
                    match_dt_idx = j
            if match_dt_idx >= 0:
                gt_match[i] = match_dt_idx
                dt_match[match_dt_idx] = i

        match_tuples = []

        # calculate distances
        for i, gt in enumerate(gts):
            gt_text = gt['text']
            if gt_match[i] >= 0:
                # matched GT
                dt_text = dts[gt_match[i]]['text']
            else:
                # unmatched GT
                dt_text = u''
            dist = text_distance(gt_text, dt_text)
            #print(i, dist)
            example_dist += dist
            match_tuples.append((gt_text, dt_text, dist))

        for i, dt in enumerate(dts):
            if dt_match[i] == -1:
                # unmatched DT
                gt_text = u''
                dt_text = dts[i]['text']
                dist = text_distance(gt_text, dt_text)
                #print(i,dist)
                example_dist += dist
                match_tuples.append((gt_text, dt_text, dist))

        # accumulate distance
        print(example_dist)
        total_dist += example_dist


    average_dist = total_dist / num_test
    print('Average distance: %d / %d = %f' % (total_dist, len(test_gt_files), average_dist))
    return average_dist




if __name__ == "__main__":
    from UDUP_pp.Allconfig.Path_Config import rec_eval_gt,rec_eval_rec
    recog_eval(rec_eval_gt,rec_eval_rec)
