// word-cloud.js

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
