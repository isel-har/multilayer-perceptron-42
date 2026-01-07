#include "visualizer.hpp"

    // Helper to generate the X-axis (epochs) based on data size
std::vector<double> Visualizer::get_epochs(size_t size) {
    std::vector<double> epochs(size);
    std::iota(epochs.begin(), epochs.end(), 1.0); // 1.0, 2.0, ...
    return epochs;
}

void Visualizer::plot_metric(const std::string& title, const std::vector<double>& data, 
                            const std::string& ylabel, const std::string& color)
{  
    std::vector<double> epochs = get_epochs(data.size());

    plt::figure();
    plt::named_plot(title, epochs, data, color);
    plt::title(title);
    plt::xlabel("Epoch");
    plt::ylabel(ylabel);
    plt::grid(true);
    plt::legend();
}

void Visualizer::double_plot_metric(const std::string& title, 
    std::pair<const std::vector<double>&, std::vector<double>&> data, 
    const std::string& ylabel, const std::string& color)
{
    std::vector<double> epochs = get_epochs(data.first.size());
    plt::figure();
    plt::named_plot(title, epochs, data.first, color);
    plt::named_plot(title, epochs, data.second, color); // to add!
    plt::title("Training " + title);
    plt::xlabel(ylabel);
    plt::ylabel(title);
    plt::grid(true);
}

void Visualizer::show() {
    plt::show();
}
