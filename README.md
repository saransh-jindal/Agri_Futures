# ML Pipeline for Agricultural Futures Prediction

A machine learning pipeline that combines NDVI (Normalized Difference Vegetation Index) data with futures market data to predict agricultural derivatives price movements.

## Features

- Data processing for NDVI and futures data
- Hybrid CNN-LSTM model architecture
- Automated training pipeline
- Real-time prediction system
- Interactive dashboard
- Automated scheduling system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-pipeline.git
cd ml-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the installation script:
```bash
sudo ./scripts/install.sh
```

## Usage

1. Configure the pipeline:
   - Edit `config/pipeline.json` for pipeline settings
   - Edit `config/scheduler.json` for scheduling settings

2. Start the service:
```bash
sudo systemctl start ml-scheduler
```

3. Access the dashboard:
   - Open your browser and navigate to `http://localhost:8050`

## Project Structure

```
ml-pipeline/
├── src/
│   └── modeling/
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       ├── predict.py
│       ├── evaluate.py
│       ├── dashboard.py
│       └── scheduler.py
├── config/
│   ├── pipeline.json
│   └── scheduler.json
├── scripts/
│   ├── install.sh
│   ├── uninstall.sh
│   ├── deploy.sh
│   └── rollback.sh
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
