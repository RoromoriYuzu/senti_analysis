# comments.py

import re
import pandas as pd
from datetime import datetime
import requests


def rescale_integer(value):
    if value == 1:
        return 10
    elif value == 2:
        return 6
    elif value == 3:
        return 3
    else:
        raise ValueError("Original value must be in range [1, 3]")


def get_ly_comments(poi_id, page_size):
    i = 0
    commentList = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    while True:
        i = i + 1
        print('正在抓取同程旅游第', i, '页评论...')
        url = f'https://www.ly.com/scenery/AjaxHelper/DianPingAjax.aspx?action=GetDianPingList&sid={poi_id}&page={i}&pageSize={page_size}&labId=1'
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            comments = response.json()['dpList']
            if comments is None:
                print('没有更多评论.')
                break
            for comment in comments:
                user_id = comment.get('dpId')
                user_nick = comment.get('dpUserName')
                score = rescale_integer(int(comment.get('servicePoint', None)))
                content = comment.get('dpContent', None).replace('\n', ' ')
                ip_address = comment.get('DPLocation', "未知")
                if not ip_address:
                    ip_address = "未知"
                publish_time = int(datetime.strptime(comment.get('dpDate', ''), "%Y-%m-%d").timestamp())
                commentList.append([publish_time, ip_address, str(user_id), user_nick, content, score])
        else:
            print('Error:', response.content)
            break
    return commentList


def get_ctrip_comments(poi_id, page_size):
    commentList = []
    url = 'https://m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    i = 0
    while True:
        i += 1
        print('正在抓取携程旅游第', i, '页评论...')
        data = {"arg": {
            "channelType": 2, "collapseType": 0, "commentTagId": 0,
            "pageIndex": i, "pageSize": page_size, "poiId": poi_id,
            "sourceType": 1, "sortType": 3, "starType": 0}}
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            comments = response.json()['result']['items']
            if comments is None:
                print('没有更多评论.')
                break
            for comment in comments:
                user_info = comment.get('userInfo', {})
                user_id = user_info.get('userId') if user_info else None
                user_nick = user_info.get('userNick') if user_info else None
                score = comment.get('score', "")
                content = comment.get('content', "").replace('\n', ' ')
                ip_address = comment.get('ipLocatedName', "未知")
                if not ip_address:
                    ip_address = "未知"
                publish_time_str = comment.get('publishTime', '')
                timestamp_match = re.search(r'\d+', publish_time_str) if publish_time_str else None
                publish_time = int(timestamp_match.group()) / 1000 if timestamp_match else None
                commentList.append([publish_time, ip_address, str(user_id), user_nick, content, int(score)])
        else:
            print('Error:', response.content)
            break
    return commentList


def get_comments(ly_poi_id, ctrip_poi_id, page_size=50, output_path='./output/Comments.csv'):
    ly_comments = get_ly_comments(ly_poi_id, page_size)
    ctrip_comments = get_ctrip_comments(ctrip_poi_id, page_size)

    columns = ['评论时间', 'IP', '用户ID', '昵称', '评论', '评分']
    df = pd.DataFrame(ly_comments + ctrip_comments, columns=columns)
    df.to_csv(output_path, index=False, encoding='utf-8')

    return output_path
