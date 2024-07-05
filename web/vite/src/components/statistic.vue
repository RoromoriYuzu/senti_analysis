<template>
   <el-card  shadow="hover">
    <el-row>
      <el-col  :span="5">
        <el-statistic title="总评论数" :value="allCommentOpt" />
      </el-col>
      <el-col :span="5">
        <el-statistic title="好评数" :value="goodOpt" :value-style="{ color: '#67c23a' }" />
      </el-col>
      <el-col :span="4">
        <el-statistic title="中评数" :value="midOpt" :value-style="{ color: '#e6a23c' }" />
      </el-col>
      <el-col :span="5">
        <el-statistic title="差评数" :value="badOpt" :value-style="{ color: '#f56c6c' }" />
      </el-col>
      <el-col :span="5">
        <el-statistic title="包含省份" :value="allIpCommentOpt" />
      </el-col>
    </el-row>
  </el-card>
  </template>
  
  <script setup>
  import { ref } from 'vue'
  import { useTransition } from '@vueuse/core'
  import { ChatLineRound, Male } from '@element-plus/icons-vue'
  import { getCmtCount, getIPCount, getCmtCountBySentiment } from '@/api/count'

  const allComment = ref(0)
  const allCommentCount = async () => allComment.value = (await getCmtCount()).count
  const allCommentOpt = useTransition(allComment, { duration: 1500 })
  allCommentCount()

  const allIpComment = ref(0)
  const allIpCommentCount = async () => allIpComment.value = (await getIPCount()).count
  const allIpCommentOpt = useTransition(allIpComment, { duration: 1500 })
  allIpCommentCount()

  const good = ref(0)
  const goodCount = async () => good.value = (await getCmtCountBySentiment(1)).count
  const goodOpt = useTransition(good, { duration: 1500 })
  goodCount()

  const mid = ref(0)
  const midCount = async () => mid.value = (await getCmtCountBySentiment(0)).count
  const midOpt = useTransition(mid, { duration: 1500 })
  midCount()

  const bad = ref(0)
  const badCount = async () => bad.value = (await getCmtCountBySentiment(-1)).count
  const badOpt = useTransition(bad, { duration: 1500 })
  badCount()

</script>
  
  <style scoped>
  .el-col {
    text-align: center;
  }

  .el-col-4-8 {
    width: 20%;
  }

</style>
  