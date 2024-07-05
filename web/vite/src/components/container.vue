<template>
  <statistic/>
    <el-row>
      <el-col  :span="6">
        <el-card shadow="hover" class="card" >
          <template #header>
            <span  class="cardHeader" >评论分布情感饼状图</span>
          </template>
          <div id="sentiment-pie-chart"></div>
        </el-card>
      </el-col>

      <el-col  :span="12">
        <el-card shadow="hover"  class="card" >
          <template #header>
            <span  class="cardHeader" >时间分布折线图</span>
          </template>
          <div id="time-line-chart"></div>
        </el-card>
      </el-col>

      <el-col  :span="6">
        <el-card shadow="hover"  class="card" >
          <template #header>
            <span  class="cardHeader" >评论最多的前五个地区饼状图</span>
          </template>
          <div id="ip-pie-chart"></div>
        </el-card>
      </el-col>
    </el-row>

      <el-card shadow="hover" class="card" >
        <template #header>
          <span  class="cardHeader" >评论数据</span>
        </template>
        <template #default>
          <el-row>
            <el-col :span="12">
              <el-card shadow="hover" class="card" >
                  <template #header>
                    <span  class="cardHeader" >好评词云</span>
                  </template>
                <div id="goodWordCloud"></div>
              </el-card>
            </el-col>
            <el-col :span="12">
              <el-card shadow="hover" class="card" >
                  <template #header>
                    <span  class="cardHeader" >差评词云</span>
                  </template>
                <div id="badWordCloud"></div>
              </el-card>
            </el-col>
          </el-row>
          <el-card>
            <template #header>
              <span class="cardHeader">随机评论</span>
            </template>
              <el-card shadow="hover" class="commentCard">
                <el-space direction="vertical">
                    <el-avatar :icon="UserFilled"/>
                    <el-text>
                      {{ comment.昵称 }}
                    </el-text>
                    <br />
                    <el-rate 
                      v-model="rate"
                      :colors="colors"
                      size="large" 
                      disabled
                      :max=5 />
                    <br />
                    <el-text>
                      {{ comment.评论 }}
                    </el-text>
                </el-space>
              </el-card>
          </el-card>

          <!-- <el-table :data="tableData" style="width: 100%">
            <el-table-column prop="IP" label="地址" width="180" />
            <el-table-column prop="昵称" label="昵称" width="180" />
            <el-table-column prop="评论" label="评论" />
          </el-table> -->

        </template>

      </el-card>

      <el-card>
        <template #header>
          <span class="cardHeader">在线情感分析</span>
        </template>
        <template #default>
          <el-card shadow="hover" class="result" >
            <template #header>
              <span  class="cardHeader" >返回结果</span>
            </template>
            <template #default>
                  <el-card style="width: 300px;">
                    <div class="progress-section">
                      <p>逻辑回归正向概率:</p>
                      <el-progress
                        :percentage="analysisResults.lr_probabilities[1] * 100"
                        :format="percentFormat"
                      ></el-progress>
                    </div>
                    <div class="progress-section">
                      <p>神经网络正向概率:</p>
                      <el-progress
                        :percentage="analysisResults.nn_probabilities[1] * 100"
                        :format="percentFormat"
                      ></el-progress>
                    </div>
                    <div class="progress-section">
                      <p>组合正向概率:</p>
                      <el-progress
                        :percentage="analysisResults.combined_prob[1] * 100"
                        :format="percentFormat"
                      ></el-progress>
                    </div>
                    <div class="sentiment-result">
                      <p>情感标签: {{ analysisResults.sentiment_label === 1 ? '正面' : '负面' }}</p>
                    </div>
                  </el-card>
            </template>
          </el-card>

          <div class="analysis">
            <el-input
            style="width: 400px"
            v-model="text"
            minlength="20"
            maxlength="200"
            placeholder="输入要分析的文本"
            :autosize="{ minRows: 2, maxRows: 5 }"
            show-word-limit
            type="textarea"
          />
            <el-button style="margin-top: 20px;" type="primary" @click="getSentimentValue" :loading=isLoading>分析</el-button>
          </div>

        </template>

      </el-card>


</template>

<script setup>
import { ref } from 'vue'
import * as d3 from 'd3'
import statistic from './statistic.vue';
import { getSentimentList, getIPList, getCmtCountByTime } from '@/api/count'
import { getKeywords, getKeywordBySentiment } from '@/api/keyword'
import { getRandomComments, getSentiment } from '@/api/comment'
import { drawPieChart, drawTimeLineChart, drawWordCloud } from '@/utils/chartUtils';
import { UserFilled } from '@element-plus/icons-vue'

const colors = ref(['#99A9BF', '#F7BA2A', '#FF9900'])

const sentimentData = ref([]);
(async () => {
  sentimentData.value = await getSentimentList();
  const data = Object.entries(sentimentData.value).map(([key, value]) => ({
    key: key === '1' ? '好评' : key === '0' ? '中评' : '差评',
    value: value
  }));
  drawPieChart('#sentiment-pie-chart', data, 330, 330);
})();



const ipData = ref([]);
(async () => {
  ipData.value = await getIPList(5);
  const data = Object.entries(ipData.value).map(([key, value]) => ({
    key: key,
    value: value
  }));
  drawPieChart('#ip-pie-chart', data, 330, 330);
})();

const timeData = ref([]);
(async () => {
  timeData.value = await getCmtCountByTime();
  const data = Object.entries(timeData.value).map(([key, value]) => ({
    key: key,
    value: value
  }));
  const width = 700;
  const height = 330;
  drawTimeLineChart('#time-line-chart', data, width, height);
})();


const goodWordCloudData = ref([]);
(async () => {
  goodWordCloudData.value = await getKeywordBySentiment(1, 80);
  const width = 600;
  const height = 330;
  drawWordCloud('#goodWordCloud', goodWordCloudData.value, width, height);
})();

const badWordCloudData = ref([]);
(async () => {
  badWordCloudData.value = await getKeywordBySentiment(-1, 80);
  const width = 600;
  const height = 330;
  drawWordCloud('#badWordCloud', badWordCloudData.value, width, height);
})();

const comment = ref('暂无评论');
let rate = ref(0);
(async function loop() {
  while (true) {
    const comments = await getRandomComments();
    comment.value = comments;
    rate.value = comments.评分 / 2;
    await new Promise(resolve => setTimeout(resolve, 3000));
  }
})();

let isLoading = ref(false)
const text = ref('')
const analysisResults = ref({
  combined_prob: [0, 0],
  lr_probabilities: [0, 0],
  nn_probabilities: [0, 0],
  sentiment_label: -1
});
const getSentimentValue = async () => {
  isLoading.value = true
  analysisResults.value = await getSentiment(text.value)
  isLoading.value = false
}

// 模拟的情感分析结果


// 格式化百分比显示
const percentFormat = (percentage) => {
  return `${parseFloat(percentage).toFixed(2)}%`;
};

</script>

<style scoped>
.cardHeader {
  display: flex;
  justify-content: center;
  font-size: 20px;
}

#goodWordCloud,
#badWordCloud {
  display: flex;
  justify-content: center;
}

.commentCard {
  margin: 0% auto;
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
  width: 400px;
  height: 400px;
}

.analysis {
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
  margin: 20px auto;
}

.result {
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
  margin: 20px auto;
  width: 500px;
}
</style>