<template>
    <div class="common-layout">
      <el-container>
        <el-header>
            <headers />
    </el-header>
        <el-main>
            <container />
        </el-main>
      </el-container>
    </div>
  </template>
  
  <script setup>
  import headers from './components/header.vue'
  import container from './components/container.vue'

</script>
  

  