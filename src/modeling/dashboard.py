#!/usr/bin/env python3

import os
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from .predict import Predictor

logger = logging.getLogger(__name__)

class Dashboard:
    """Dashboard for visualizing model results and predictions."""
    
    def __init__(self, config):
        """Initialize the dashboard with configuration."""
        self.config = config
        self.predictor = Predictor(config)
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            html.H1('Agricultural Futures Prediction Dashboard'),
            
            # Prediction Section
            html.Div([
                html.H2('Latest Predictions'),
                dcc.Graph(id='prediction-graph'),
                html.Div(id='prediction-stats')
            ]),
            
            # Historical Performance
            html.Div([
                html.H2('Historical Performance'),
                dcc.Graph(id='performance-graph')
            ]),
            
            # Update Interval
            dcc.Interval(
                id='interval-component',
                interval=self.config['refresh_interval'] * 1000,  # in milliseconds
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Set up dashboard callbacks."""
        @self.app.callback(
            [Output('prediction-graph', 'figure'),
             Output('prediction-stats', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_predictions(n):
            try:
                # Get latest predictions
                results = self.predictor.predict_latest()
                
                # Create prediction figure
                fig = go.Figure()
                
                # Add prediction line
                fig.add_trace(go.Scatter(
                    x=results['dates'],
                    y=results['predictions'].flatten(),
                    name='Prediction',
                    line=dict(color='blue')
                ))
                
                # Add confidence intervals
                fig.add_trace(go.Scatter(
                    x=results['dates'],
                    y=results['confidence'].flatten(),
                    name='Confidence',
                    line=dict(color='gray', dash='dash')
                ))
                
                # Update layout
                fig.update_layout(
                    title='Price Movement Predictions',
                    xaxis_title='Date',
                    yaxis_title='Probability',
                    hovermode='x unified'
                )
                
                # Calculate statistics
                latest_pred = results['predictions'][-1][0]
                latest_conf = results['confidence'][-1][0]
                latest_dir = results['direction'][-1][0]
                
                stats = html.Div([
                    html.H3('Latest Prediction'),
                    html.P(f'Direction: {latest_dir}'),
                    html.P(f'Probability: {latest_pred:.2%}'),
                    html.P(f'Confidence: {latest_conf:.2%}')
                ])
                
                return fig, stats
                
            except Exception as e:
                logger.error(f"Error updating predictions: {str(e)}")
                return {}, html.Div('Error updating predictions')
                
        @self.app.callback(
            Output('performance-graph', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_performance(n):
            try:
                # Get historical performance
                results = self.predictor.predict_latest()
                
                # Create performance figure
                fig = go.Figure()
                
                # Add accuracy line
                fig.add_trace(go.Scatter(
                    x=results['dates'],
                    y=results['predictions'].flatten(),
                    name='Accuracy',
                    line=dict(color='green')
                ))
                
                # Update layout
                fig.update_layout(
                    title='Model Performance',
                    xaxis_title='Date',
                    yaxis_title='Accuracy',
                    hovermode='x unified'
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating performance: {str(e)}")
                return {}
                
    def run(self):
        """Run the dashboard."""
        try:
            self.app.run_server(
                host=self.config['host'],
                port=self.config['port'],
                debug=False
            )
            
        except Exception as e:
            logger.error(f"Error running dashboard: {str(e)}")
            raise

def main():
    """Main function to run the dashboard."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and run dashboard
    dashboard = Dashboard(
        config={
            'host': '0.0.0.0',
            'port': 8050,
            'refresh_interval': 5  # minutes
        }
    )
    dashboard.run()

if __name__ == "__main__":
    main() 
