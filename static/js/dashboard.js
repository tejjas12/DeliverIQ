
"use strict";

// ==========================================
// DELIVERIQ - FOOD DELIVERY ANALYTICS
// Dashboard + ML Models + Prediction
// ==========================================


// CHART CONFIGURATION

const chartLayout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",

    font: {
        color: "#cbd5e1",
        family: "Inter"
    },

    margin: {
        t: 20,
        r: 20,
        b: 65,
        l: 60
    },

    xaxis: {
        gridcolor: "rgba(148,163,184,0.12)",
        zerolinecolor: "rgba(148,163,184,0.12)"
    },

    yaxis: {
        gridcolor: "rgba(148,163,184,0.12)"
    },

    showlegend: false,
    autosize: true
};


const chartConfig = {
    responsive: true,
    displayModeBar: false
};


// ==========================================
// HELPER FUNCTIONS
// ==========================================

async function fetchJSON(url, options = {}) {

    const response = await fetch(url, options);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || `Request failed: ${response.status}`);
    }

    return data;
}


function showChartMessage(elementId, message) {

    const element = document.getElementById(elementId);

    if (element) {
        element.innerHTML = `
            <div style="
                display:flex;
                justify-content:center;
                align-items:center;
                min-height:200px;
                color:#94a3b8;
                font-size:13px;
                text-align:center;
            ">
                ${message}
            </div>
        `;
    }
}


// Normalize CSV column names

function normalizeColumnName(name) {

    return String(name)
        .toLowerCase()
        .replace(/[^a-z0-9]/g, "");
}


function getMetricValue(row, possibleNames) {

    const normalizedRow = {};

    Object.keys(row).forEach(key => {
        normalizedRow[normalizeColumnName(key)] = row[key];
    });

    for (const name of possibleNames) {

        const key = normalizeColumnName(name);

        if (
            Object.prototype.hasOwnProperty.call(normalizedRow, key)
        ) {
            return normalizedRow[key];
        }
    }

    return undefined;
}


// ==========================================
// 1. DASHBOARD OVERVIEW
// ==========================================

async function loadOverview() {

    try {

        const data = await fetchJSON("/api/overview");

        document.getElementById("totalRecords").textContent =
            Number(data.total_records).toLocaleString("en-IN");

        document.getElementById("avgTime").textContent =
            data.average_delivery_time ?? "--";

        document.getElementById("minTime").textContent =
            data.minimum_delivery_time ?? "--";

        document.getElementById("maxTime").textContent =
            data.maximum_delivery_time ?? "--";

    } catch (error) {

        console.error("Overview Error:", error);

        document.getElementById("totalRecords").textContent = "Error";
    }
}


// ==========================================
// 2. ANALYTICS CHART GENERATOR
// ==========================================

function createAnalyticsChart(elementId, records, categoryColumn) {

    if (!records || records.length === 0) {

        showChartMessage(elementId, "No data available");
        return;
    }

    const categories = records.map(
        row => row[categoryColumn]
    );

    const values = records.map(
        row => Number(row["Time_taken (min)"])
    );

    const trace = {

        x: categories,
        y: values,
        type: "bar",

        marker: {
            color: "#8b5cf6",
            line: {
                color: "#22d3ee",
                width: 1
            }
        },

        hovertemplate:
            "%{x}<br>Average: %{y:.2f} min<extra></extra>"
    };

    Plotly.newPlot(

        elementId,

        [trace],

        {
            ...chartLayout,

            xaxis: {
                ...chartLayout.xaxis,
                title: categoryColumn.replaceAll("_", " ")
            },

            yaxis: {
                ...chartLayout.yaxis,
                title: "Average Delivery Time (min)"
            }
        },

        chartConfig
    );
}


// ==========================================
// 3. LOAD ANALYTICS
// ==========================================

async function loadAnalytics() {

    try {

        const data = await fetchJSON("/api/analytics");

        createAnalyticsChart(
            "weatherChart",
            data.Weather_conditions,
            "Weather_conditions"
        );

        createAnalyticsChart(
            "trafficChart",
            data.Road_traffic_density,
            "Road_traffic_density"
        );

        createAnalyticsChart(
            "cityChart",
            data.City,
            "City"
        );

        createAnalyticsChart(
            "vehicleChart",
            data.Type_of_vehicle,
            "Type_of_vehicle"
        );

    } catch (error) {

        console.error("Analytics Error:", error);

        [
            "weatherChart",
            "trafficChart",
            "cityChart",
            "vehicleChart"
        ].forEach(id => {
            showChartMessage(id, "Unable to load analytics");
        });
    }
}


// ==========================================
// 4. MACHINE LEARNING MODEL PERFORMANCE
// ==========================================

async function loadModels() {

    try {

        const data = await fetchJSON("/api/models");

        const results = data.results;

        if (!results || results.length === 0) {
            throw new Error("No model results available");
        }


        // Extract model metrics

        const validResults = results.map(row => {

            const modelName = getMetricValue(
                row,
                ["Model", "Model Name", "Algorithm"]
            );

            const mae = Number(
                getMetricValue(row, ["MAE", "Mean Absolute Error"])
            );

            const rmse = Number(
                getMetricValue(row, ["RMSE", "Root Mean Squared Error"])
            );

            const r2Raw = getMetricValue(
                row,
                [
                    "R2",
                    "R²",
                    "R2 Score",
                    "R Squared",
                    "R2_Score",
                    "R_squared",
                    "R-squared"
                ]
            );

            const r2 = Number(r2Raw);

            return {
                model: modelName,
                mae: mae,
                rmse: rmse,
                r2: r2
            };

        }).filter(row =>
            row.model !== undefined &&
            Number.isFinite(row.mae)
        );


        if (validResults.length === 0) {
            throw new Error("Unable to identify model metrics");
        }


        // Lowest MAE = Best model

        const best = validResults.reduce((a, b) =>
            a.mae < b.mae ? a : b
        );


        // Update best model cards

        document.getElementById("bestModel").textContent =
            best.model;

        document.getElementById("bestMAE").textContent =
            best.mae.toFixed(2);

        document.getElementById("bestR2").textContent =
            Number.isFinite(best.r2)
                ? best.r2.toFixed(3)
                : "--";


        // Model comparison chart

        const trace = {

            x: validResults.map(row => row.model),

            y: validResults.map(row => row.mae),

            type: "bar",

            marker: {
                color: validResults.map(row =>
                    row.model === best.model ? "#22d3ee" : "#8b5cf6"
                )
            },

            hovertemplate:
                "%{x}<br>MAE: %{y:.2f} min<extra></extra>"
        };


        Plotly.newPlot(

            "modelChart",

            [trace],

            {
                ...chartLayout,

                yaxis: {
                    ...chartLayout.yaxis,
                    title: "Mean Absolute Error (min)"
                },

                xaxis: {
                    ...chartLayout.xaxis,
                    title: "Machine Learning Algorithms"
                }
            },

            chartConfig
        );


        console.log("Model evaluation loaded:", validResults);

    } catch (error) {

        console.error("Model Error:", error);

        showChartMessage(
            "modelChart",
            "Unable to load model comparison"
        );

        document.getElementById("bestR2").textContent = "--";
    }
}


// ==========================================
// 5. DELIVERY TIME PREDICTION
// ==========================================

async function handlePrediction(event) {

    event.preventDefault();

    const form = event.target;

    const button = document.getElementById("predictButton");

    const resultBox = document.getElementById("predictionResult");

    const errorBox = document.getElementById("predictionError");

    const timeElement = document.getElementById("predictedTime");

    const modelElement = document.getElementById("predictionModel");

    errorBox.textContent = "";
    resultBox.hidden = true;

    if (!form.reportValidity()) {
        return;
    }

    button.disabled = true;
    button.textContent = "Predicting... Please wait";

    try {

        const formData = new FormData(form);
        const payload = {};

        const numericFields = [
            "Delivery_person_Age",
            "Delivery_person_Ratings",
            "Vehicle_condition",
            "multiple_deliveries",
            "Delivery_Distance_km",
            "Order_Hour",
            "Order_Month",
            "Order_Weekday",
            "Is_Weekend"
        ];

        for (const [key, value] of formData.entries()) {

            if (numericFields.includes(key)) {
                payload[key] = Number(value);
            } else {
                payload[key] = value;
            }
        }

        const result = await fetchJSON(
            "/api/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(payload)
            }
        );

        if (!result.success) {
            throw new Error("Prediction was not successful");
        }

        timeElement.textContent =
            Number(result.predicted_delivery_time).toFixed(2);

        modelElement.textContent =
            "Powered by " + result.model;

        resultBox.hidden = false;

        resultBox.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

    } catch (error) {

        console.error("Prediction Error:", error);

        errorBox.textContent =
            error.message || "Unable to predict delivery time.";

    } finally {

        button.disabled = false;

        button.textContent = "✦ Predict Delivery Time";
    }
}


// ==========================================
// 6. SIDEBAR NAVIGATION
// ==========================================

function setupNavigation() {

    const links = document.querySelectorAll(".nav-link");

    links.forEach(link => {

        link.addEventListener("click", function () {

            links.forEach(item => {
                item.classList.remove("active");
            });

            this.classList.add("active");
        });
    });
}


// ==========================================
// 7. INITIALIZE DASHBOARD
// ==========================================

document.addEventListener("DOMContentLoaded", () => {

    console.log("DeliverIQ Dashboard Initialized");

    loadOverview();

    loadAnalytics();

    loadModels();

    const predictionForm = document.getElementById("predictionForm");

    if (predictionForm) {

        predictionForm.addEventListener(
            "submit",
            handlePrediction
        );
    }

    setupNavigation();

    window.addEventListener("resize", () => {

        document.querySelectorAll(".chart").forEach(chart => {

            if (chart.data) {
                Plotly.Plots.resize(chart);
            }

        });

    });

});
