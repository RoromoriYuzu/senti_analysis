import request from "@/utils/request";

//获取指定ip的评论
export function getCommentByIp(ip){
    return request.get('/getCommentByIp',{params:ip})
}

//获取指定情感的评论
export function getCmtBySentiment(sentiment){
    return request.get('/getCmtBySentiment',{params:sentiment})
}

//获取指定索引的评论
export function getComments(index,end){
    return request.get('/getComments',{params:{index,end}})
}

//获取随机评论
export function getRandomComments(){
    return request.get('/getRandomComments')
}

//获取评论的情感值
export function getSentiment(comment){
    return request.get('/getSentiment',{params:{comment:comment}})
}