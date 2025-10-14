import pandas as pd
import matplotlib.pyplot as plt
import sys


def plot_data(filename='glucose_data.csv'):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)
    # Check if the necessary columns exist
    if 'Time' not in df.columns or 'Glycemia' not in df.columns or 'Insulin Infusion' not in df.columns:
        print("CSV file must contain 'Time', 'Glycemia', and 'Insulin Infusion' columns.")
        return

    # Plot Time vs. Glucose Level with thinner lines (no markers)
    plt.figure(figsize=(10, 5))
    plt.plot(df['Time'], df['Glycemia'], color='r', label='Glycemia',
             linewidth=1.5)  # Thinner lines without markers
    plt.xlabel('Time (minutes)')
    plt.ylabel('Glycemia (mg/dL)')
    plt.title('Time vs. Glucose Level')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('glucose_level_plot.png')  # Save the plot to a file

    # Plot Time vs. Insulin Infusion Rate with thinner lines (no markers)
    plt.figure(figsize=(10, 5))
    plt.plot(df['Time'], df['Insulin Infusion'], color='b',
             label='Insulin Infusion', linewidth=1.5)  # Thinner lines without markers
    plt.xlabel('Time (minutes)')
    plt.ylabel('Insulin Infusion Rate (units/hr)')
    plt.title('Time vs. Insulin Infusion Rate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('insulin_infusion_plot.png')  # Save the plot to a file

    # Show the plots
    plt.show()


def main():
    # Check if the filename is provided as an argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'glucose_data.csv'

    plot_data(filename=filename)


if __name__ == "__main__":
    main()
