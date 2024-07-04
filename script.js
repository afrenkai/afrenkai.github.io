// Word Cloud
var words = [
    {text: "Python", size: 40},
    {text: "Machine Learning", size: 50},
    {text: "Data Analysis", size: 30},
    {text: "Neural Networks", size: 20},
    {text: "Deep Learning", size: 60},
    {text: "NLP", size: 25},
    {text: "TensorFlow", size: 35},
    {text: "Keras", size: 20},
    {text: "scikit-learn", size: 30},
    {text: "Pandas", size: 40},
    {text: "NumPy", size: 30}
];

var layout = d3.layout.cloud()
    .size([500, 300])
    .words(words.map(function(d) {
        return {text: d.text, size: d.size};
    }))
    .padding(5)
    .rotate(function() { return ~~(Math.random() * 2) * 90; })
    .font("Impact")
    .fontSize(function(d) { return d.size; })
    .on("end", draw);

layout.start();

function draw(words) {
    d3.select("#word-cloud").append("svg")
        .attr("width", layout.size()[0])
        .attr("height", layout.size()[1])
        .append("g")
        .attr("transform", "translate(" + layout.size()[0] / 2 + "," + layout.size()[1] / 2 + ")")
        .selectAll("text")
        .data(words)
        .enter().append("text")
        .style("font-size", function(d) { return d.size + "px"; })
        .style("font-family", "Impact")
        .style("fill", function(d, i) { return d3.schemeCategory10[i % 10]; })
        .attr("text-anchor", "middle")
        .attr("transform", function(d) {
            return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
        })
        .text(function(d) { return d.text; });
}

// Bar Chart
var ctx = document.getElementById('bar-chart').getContext('2d');
var myChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['Python', 'TensorFlow', 'Keras', 'scikit-learn', 'Pandas', 'NumPy'],
        datasets: [{
            label: 'Proficiency',
            data: [9, 7, 7, 8, 9, 8],
            backgroundColor: [
                'rgba(255, 99, 132, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(75, 192, 192, 0.2)',
                'rgba(153, 102, 255, 0.2)',
                'rgba(255, 159, 64, 0.2)'
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 159, 64, 1)'
            ],
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});
