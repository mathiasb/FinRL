import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import pandas as pd

class InteractiveChart:
    """
    A class to handle interactive plotting in Jupyter Notebooks.
    Ensures stable rendering by using a dedicated Output widget and clear_output logic.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        if 'tic' not in df.columns:
            raise ValueError("DataFrame must have a 'tic' column for tickers.")
            
        self.available_tickers = sorted(df['tic'].unique().tolist())
        self.available_indicators = ['rsi_30', 'macd', 'cci_30', 'dx_30', 'atr', 'boll_ub', 'boll_lb', 'close']
        
        # Colors for tickers
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        # Widgets
        self.ticker_widget = widgets.SelectMultiple(
            options=self.available_tickers,
            value=[self.available_tickers[0]] if self.available_tickers else [],
            description='Tickers',
            rows=min(len(self.available_tickers), 10)
        )
        
        self.indicator_widget = widgets.Dropdown(
            options=self.available_indicators,
            value='rsi_30',
            description='Indicator',
        )
        
        # Output Widget - The container for the plot
        self.out = widgets.Output()
        
        # Bind events
        self.ticker_widget.observe(self.plot, names='value')
        self.indicator_widget.observe(self.plot, names='value')
        
    def plot(self, change=None):
        """
        The plotting logic. Clears the output widget and rendering a new figure.
        Uses plt.ioff() to prevent auto-display of the figure before we are ready.
        """
        # Prevent matplotlib from displaying the figure immediately when created
        plt.ioff()
        
        with self.out:
            # Critical: Clear previous plot to avoid stacking
            clear_output(wait=True)
            
            tickers = self.ticker_widget.value
            indicator = self.indicator_widget.value
            
            if not tickers:
                print("Please select at least one ticker.")
                return
            
            try:
                # Create Figure
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                # --- Primary Axis (Left): Close Price ---
                lines = []
                for i, tic in enumerate(tickers):
                    subset = self.df[self.df['tic'] == tic]
                    if subset.empty:
                        continue
                    
                    # Plot Price
                    line, = ax1.plot(subset.index, subset['close'], 
                                     label=f"{tic} Price",
                                     color=self.colors[i % len(self.colors)],
                                     linestyle='-', linewidth=1.5)
                    lines.append(line)
                
                ax1.set_ylabel("Close Price", fontsize=12, fontweight='bold')
                ax1.set_title(f"Price vs {indicator.upper()}", fontsize=14)
                ax1.grid(True, alpha=0.2)
                
                # --- Secondary Axis (Right): Indicator ---
                if indicator != 'close':
                    ax2 = ax1.twinx()
                    for i, tic in enumerate(tickers):
                        subset = self.df[self.df['tic'] == tic]
                        if subset.empty:
                            continue
                            
                        if indicator in subset.columns:
                            # Plot Indicator (Dashed)
                            line, = ax2.plot(subset.index, subset[indicator], 
                                             label=f"{tic} {indicator} (R)",
                                             color=self.colors[i % len(self.colors)],
                                             linestyle='--', linewidth=1.5, alpha=0.8)
                            lines.append(line)
                    
                    ax2.set_ylabel(indicator, fontsize=12, fontweight='bold')
                
                # Consolidated Legend
                final_lines = lines
                final_labels = [l.get_label() for l in final_lines]
                ax1.legend(final_lines, final_labels, loc='upper left', 
                           frameon=True, fancybox=True, shadow=True)
                
                # Explicitly display the figure
                display(fig)
                
            except Exception as e:
                print(f"Error plotting: {e}")
            finally:
                # Ensure figure is closed to release memory and prevent double-display
                plt.close(fig)
                # Restore interactive mode defaults just in case
                plt.ion()
            
    def show(self):
        """Display the UI components."""
        # Initial Plot
        self.plot()
        
        # Layout
        ui = widgets.HBox([self.ticker_widget, self.indicator_widget])
        display(widgets.VBox([ui, self.out]))
