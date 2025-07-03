document.addEventListener('DOMContentLoaded', function () {
  document.getElementById('inputFile').addEventListener('change', function () {
    const fileName = this.files[0]?.name || 'No file selected';
    document.getElementById('fileName').textContent = `Selected: ${fileName}`;
  });

  document.getElementById('processForm').addEventListener('submit', function (e) {
    e.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('inputFile');
    formData.append('file', fileInput.files[0]);

    document.getElementById('loadingSpinner').style.display = 'block';

    fetch('/process', {
      method: 'POST',
      body: formData
    })
      .then(res => res.json())
      .then(handleResponse)
      .catch(err => showError(err.message))
      .finally(() => {
        document.getElementById('loadingSpinner').style.display = 'none';
      });
  });

  // Expose sample loader for button onclick usage
  window.loadSampleFile = function (filename) {
    document.getElementById('loadingSpinner').style.display = 'block';
    fetch(`/static/samples/${filename}`)
      .then(res => res.blob())
      .then(blob => {
        const file = new File([blob], filename, { type: 'text/csv' });
        const formData = new FormData();
        formData.append('file', file);
        return fetch('/process', {
          method: 'POST',
          body: formData
        });
      })
      .then(res => res.json())
      .then(handleResponse)
      .catch(err => showError(err.message))
      .finally(() => {
        document.getElementById('loadingSpinner').style.display = 'none';
      });
  };

  function handleResponse(data) {
    const predictionOutput = document.getElementById('predictionOutput');
    const container = document.getElementById('plotContainer');
    const row = document.createElement('div');
    row.className = 'row g-4';

    container.innerHTML = '';
    predictionOutput.innerHTML = '';

    if (data.error) {
      showError(data.error);
      return;
    }

    if (data.breakdowns?.length > 0) {
      data.breakdowns.forEach(b => {
        const col = document.createElement('div');
        col.className = 'col-md-6 fade-in';
        col.innerHTML = `
          <div class="card bg-dark text-light border-info shadow-sm h-100">
            <div class="card-body">
              <h5 class="card-title text-info">Broken Detected</h5>
              <ul class="list-group list-group-flush mt-3">
                <li class="list-group-item bg-transparent text-light d-flex justify-content-between">
                  <strong>Initial</strong><span class="text-warning">${b.start || 'N/A'}</span>
                </li>
                <li class="list-group-item bg-transparent text-light d-flex justify-content-between">
                  <strong>Final</strong><span class="text-warning">${b.end || 'N/A'}</span>
                </li>
                <li class="list-group-item bg-transparent text-light d-flex justify-content-between">
                  <strong>Duration</strong><span class="badge bg-info text-dark">${b.duration || 'N/A'}</span>
                </li>
              </ul>
            </div>
          </div>
        `;
        row.appendChild(col);
      });
    }

    if (data.recoveries?.length > 0) {
      data.recoveries.forEach(r => {
        const col = document.createElement('div');
        col.className = 'col-md-6 fade-in';
        col.innerHTML = `
          <div class="card bg-dark text-light border-success shadow-sm h-100">
            <div class="card-body">
              <h5 class="card-title text-success">Recovering Detected</h5>
              <ul class="list-group list-group-flush mt-3">
                <li class="list-group-item bg-transparent text-light d-flex justify-content-between">
                  <strong>Initial</strong><span class="text-success">${r.start || 'N/A'}</span>
                </li>
                <li class="list-group-item bg-transparent text-light d-flex justify-content-between">
                  <strong>Final</strong><span class="text-success">${r.end || 'N/A'}</span>
                </li>
                <li class="list-group-item bg-transparent text-light d-flex justify-content-between">
                  <strong>Duration</strong><span class="badge bg-success text-dark">${r.duration || 'N/A'}</span>
                </li>
              </ul>
            </div>
          </div>
        `;
        row.appendChild(col);
      });
    }

    if (row.children.length > 0) {
      predictionOutput.appendChild(row);
    } else if (data.message) {
      predictionOutput.innerHTML = `<div class="alert alert-success fade-in" role="alert">${data.message}</div>`;
    }

    if (data.plot_urls && data.fft_urls) {
      for (let i = 0; i < data.plot_urls.length; i++) {
        const sensorNumber = i + 1;
        const row = document.createElement('div');
        row.className = 'plot-pair mb-4 fade-in';

        const sensorPlot = document.createElement('div');
        sensorPlot.className = 'plot-box';
        sensorPlot.innerHTML = `
          <h6>Sensor ${sensorNumber} – Time Series</h6>
          <img src="${data.plot_urls[i]}" class="img-fluid rounded border shadow">
        `;

        const fftPlot = document.createElement('div');
        fftPlot.className = 'plot-box';
        fftPlot.innerHTML = `
          <h6>Sensor ${sensorNumber} – FFT Spectrum</h6>
          <img src="${data.fft_urls[i]}" class="img-fluid rounded border shadow">
        `;

        row.appendChild(sensorPlot);
        row.appendChild(fftPlot);
        container.appendChild(row);
      }
    }
  }

  function showError(message) {
    document.getElementById('errorToastBody').textContent = message;
    const toast = new bootstrap.Toast(document.getElementById('errorToast'));
    toast.show();
  }
});
