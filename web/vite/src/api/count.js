import request from "@/utils/request.js";

// 获取评论数量
export function getCmtCount() {
  return request.get('/getCmtCount')
}

//获取IP归属地的数量
export function getIPCount() {
  return request.get('/getIPCount')
}

// 获取指定IP归属地的评论数量
export function getCmtCountByIP(ip) {
  return request.get('/getCmtCountByIP',{params:ip})
}

// 获取IP归属地列表，以及对应的评论数量
export function getIPList(top=10) {
    return request.get('/getIPList',{params:{top:top}})
}

// 获取情感列表，以及对应的评论数量
export function getSentimentList() {
    return request.get('/getSentimentList')
}

// 获取指定情感的评论数量
export function getCmtCountBySentiment(sentiment) {
    return request.get('/getCmtCountBySentiment',{params:{sentiment:sentiment}})
}

// 获取不同时间的评论数量
export function getCmtCountByTime() {
    return request.get('/getCmtCountByTime')
}
