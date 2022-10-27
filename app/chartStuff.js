const labels = [
    'Angry',
    // 'Disgust',
    // 'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
];

const emojis = [
    0x1F621,
    // 0x1F922,
    // 0x1F628,
    0x1F603,
    0x1F610,
    0x1F62D,
    0x1F632
];

function initializeChart() {
    const ctx = document.getElementById('prediction');

    return new Chart(ctx, {
        type: 'doughnut',
        data: {},
        plugins: [ChartDataLabels],
        options: {
            legend: {
                display: false,
            },
            plugins: {
                datalabels: {
                    display: true,
                    formatter: (val, ctx) => {
                        // Grab the label for this value
                        const label = labels[ctx.dataIndex];
                        // Put them together
                        return label;
                    },
                    color: '#fff',
                    backgroundColor: '#404040',
                    font: {
                        size: 18,
                    }

                },
            },
        }

    })

}

function updateChart(chart, predictions) {


    const datasets = [{
        labels: labels,
        data: predictions,
        backgroundColor: [
            'rgb(102,194,165)',
            'rgb(252,141,98)',
            'rgb(141,160,203)',
            'rgb(231,138,195)',
            'rgb(166,216,84)',
            'rgb(255,217,47)',
            'rgb(229,196,148)'
        ]
        ,
        hoverOffset: 4
    }]

    chart.data.datasets = datasets;
    chart.update();

}

function updateEmoji(maxPrediction){
    let element = document.getElementById("emoji");
    element.textContent = String.fromCodePoint( emojis[maxPrediction])
}
