import request from "@/utils/request.js";

//获取评论的关键词
export function getKeywords(topK){
    return request.get('/getKeyword',{params:{topK}})
}

//获取指定情感的关键词
export function getKeywordBySentiment(sentiment,topK){
    return request.get('/getKeywordBySentiment',{params:{sentiment,topK}})
}