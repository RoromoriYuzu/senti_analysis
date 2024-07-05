import axios from "axios";

const bashUrl = "/api";
const request = axios.create({baseURL: bashUrl});

// 响应拦截器
request.interceptors.response.use(
    res=>{
        return res.data;
    }
),err=>{
    console.log("请求异常，请稍后再试");
    return Promise.reject(err);
}

export default request;