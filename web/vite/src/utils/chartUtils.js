// chartUtils.js
import * as d3 from 'd3';
import cloud from 'd3-cloud';



export function drawPieChart(selector, data, width, height) {
  const radius = Math.min(width, height) / 2;
  const total = data.reduce((sum, item) => sum + item.value, 0);

  const pie = d3.pie().value(d => d.value);

  const arc = d3.arc().innerRadius(0).outerRadius(radius);

  const color = d3.scaleOrdinal()
    .domain(data.map(d => d.key))
    .range(d3.schemeAccent);

  const svg = d3.select(selector)
    .append('svg')
    .attr('width', width)
    .attr('height', height)
    .append('g')
    .attr('transform', `translate(${width / 2}, ${height / 2})`);

  svg.selectAll('path')
    .data(pie(data))
    .enter()
    .append('path')
    .attr('d', arc)
    .attr('fill', d => color(d.data.key))
    .attr('stroke', 'white')
    .style('stroke-width', '2px')
    .style('opacity', 0.7);

  const labelArc = d3.arc().innerRadius(radius / 2).outerRadius(radius / 2);

  svg.selectAll('text')
    .data(pie(data))
    .enter()
    .append('text')
    .attr('transform', d => `translate(${labelArc.centroid(d)})`)
    .attr('dy', '-1px')
    .attr('text-anchor', 'middle')
    .style('fill', 'black')
    .style('font-size', '15px')
    .selectAll('tspan')
    .data(d => [`${(d.data.value / total * 100).toFixed(1)}%`, `${d.data.key}`])
    .enter()
    .append('tspan')
    .attr('x', 0)
    .attr('y', (d, i) => `${i * 1.2}em`)
    .text(d => d);
}


export function drawTimeLineChart(selector, data, width, height) {
    const margin = { top: 10, right: 30, bottom: 30, left: 60 },
          innerWidth = width - margin.left - margin.right,
          innerHeight = height - margin.top - margin.bottom;
  
    const svg = d3.select(selector)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);
  
    data.forEach(d => {
      d.key = d3.timeParse("%Y-%m")(d.key);
      d.value = +d.value;
    });
  
    // Add X axis
    const x = d3.scaleTime()
      .domain(d3.extent(data, d => d.key))
      .range([0, innerWidth]);
    const xAxis = svg.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x));
  
    // Add Y axis
    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.value)])
      .range([innerHeight, 0]);
    const yAxis = svg.append("g")
      .call(d3.axisLeft(y));
  
    // Add the line
    const line = d3.line()
      .x(d => x(d.key))
      .y(d => y(d.value));
  
    svg.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-width", 1.5)
      .attr("d", line);
  
    // Add the brushing
    const brush = d3.brushX()
      .extent([[0, 0], [innerWidth, innerHeight]])
      .on("end", updateChart);
  
    // Add the brushing
    var idleTimeout = null;
    const idleDelay = 350;
    svg.append("g")
      .attr("class", "brush")
      .call(brush);
  
    // A function that set idleTimeOut to null
    const idled = () => { idleTimeout = null; };
  
    // A function that update the chart for given boundaries
    function updateChart(event) {
      const extent = event.selection;
      if (!extent) {
        if (!idleTimeout) return idleTimeout = setTimeout(idled, idleDelay);
        x.domain([4, 8]);
      } else {
        x.domain([x.invert(extent[0]), x.invert(extent[1])]);
        svg.select(".brush").call(brush.move, null);
      }
      xAxis.transition().duration(1000).call(d3.axisBottom(x));
      svg.select(".line")
        .transition()
        .duration(1000)
        .attr("d", line);
    }
  
    // If the user double clicks, reinitialize the chart
    svg.on("dblclick", () => {
      x.domain(d3.extent(data, d => d.key));
      xAxis.transition().call(d3.axisBottom(x));
      svg.select(".line")
        .transition()
        .attr("d", line);
    });
  }


export function drawWordCloud(selector, wordsData, width = 500, height = 500) {
    // 将传入的数据转换为词云所需的格式
    const words = wordsData.map(d => ({ text: d[0], size: d[1] * 300 }));

    // 创建一个词云布局
    const layout = cloud()
        .size([width, height])
        .words(words)
        .padding(5)
        .rotate(0)
        .font('Impact')
        .fontSize(d => d.size)
        .on('end', draw);

    // 开始布局计算
    layout.start();

    // 绘制词云的函数
    function draw(words) {
        const svg = d3.select(selector)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', `translate(${width / 2},${height / 2})`);

        svg.selectAll('text')
            .data(words)
            .join('text')
            .style('font-size', d => `${d.size}px`)
            .style('font-family', 'Impact')
            .attr('text-anchor', 'middle')
            .attr('transform', d => `translate(${d.x}, ${d.y}) rotate(${d.rotate})`)
            .text(d => d.text)
            .style('fill', (d, i) => d3.schemeTableau10[i % 10]);
    }
}
