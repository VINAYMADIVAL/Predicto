import { Play, Download, AlertCircle, FileText, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";

const PrototypeDemo = () => {
  const sampleInput = [
    { field: "Project ID", value: "PG-2024-SIH-001" },
    { field: "Tower Type", value: "400kV Double Circuit" }, 
    { field: "Location", value: "Karnataka, India" },
    { field: "Budget", value: "₹52 Crores" },
    { field: "Timeline", value: "18 months" }
  ];

  const sampleOutput = [
    { material: "Steel (tons)", forecasted: "1,450", confidence: "94.2%" },
    { material: "Cement (tons)", value: "920", confidence: "91.8%" },
    { material: "Copper Wire (km)", value: "180", confidence: "89.5%" },
    { material: "Insulators (units)", value: "2,400", confidence: "92.1%" }
  ];

  return (
    <section className="py-24 bg-muted/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-flex items-center space-x-2 bg-secondary/10 backdrop-blur-sm border border-secondary/20 rounded-full px-4 py-2 mb-6">
            <Play className="w-4 h-4 text-secondary" />
            <span className="text-sm text-secondary">Live Demo</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-bold text-gradient mb-6">
            Prototype Demo
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto mb-4">
            This prototype was developed under SIH 2025. Forecasting accuracy will improve as we integrate POWERGRID's historical data.
          </p>
          
          {/* Disclaimer Banner */}
          <div className="inline-flex items-center space-x-2 bg-warning/10 backdrop-blur-sm border border-warning/20 rounded-lg px-4 py-3 mb-8">
            <AlertCircle className="w-5 h-5 text-warning" />
            <span className="text-sm text-warning font-medium">Sample data for demonstration only</span>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          {/* Sample Input */}
          <div className="glass-card p-8 animate-slide-up">
            <div className="flex items-center space-x-3 mb-6">
              <FileText className="w-6 h-6 text-primary" />
              <h3 className="text-xl font-semibold text-foreground">Sample Input CSV</h3>
            </div>
            <div className="bg-muted/30 rounded-lg p-4 mb-6">
              <div className="text-xs font-mono text-muted-foreground mb-2">project_input.csv</div>
              <div className="space-y-2">
                {sampleInput.map((item, index) => (
                  <div key={index} className="flex justify-between py-1">
                    <span className="text-sm text-muted-foreground">{item.field}:</span>
                    <span className="text-sm font-medium text-foreground">{item.value}</span>
                  </div>
                ))}
              </div>
            </div>
            <Button variant="outline" size="sm" className="w-full">
              <Download className="w-4 h-4 mr-2" />
              Download Sample CSV
            </Button>
          </div>

          {/* Sample Output */}
          <div className="glass-card p-8 animate-slide-up">
            <div className="flex items-center space-x-3 mb-6">
              <TrendingUp className="w-6 h-6 text-success" />
              <h3 className="text-xl font-semibold text-foreground">Forecast Output</h3>
            </div>
            <div className="space-y-4 mb-6">
              {sampleOutput.map((item, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                  <div>
                    <div className="text-sm font-medium text-foreground">{item.material}</div>
                    <div className="text-lg font-bold text-accent">{item.forecasted || item.value}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-muted-foreground">Confidence</div>
                    <div className="text-sm font-semibold text-success">{item.confidence}</div>
                  </div>
                </div>
              ))}
            </div>
            <Button variant="cta" size="sm" className="w-full">
              <Play className="w-4 h-4 mr-2" />
              Run Forecast Demo
            </Button>
          </div>
        </div>

        {/* SIH 2025 Badge */}
        <div className="text-center">
          <div className="inline-flex items-center space-x-3 glass-card px-6 py-4">
            <div className="w-12 h-12 bg-gradient-primary rounded-full flex items-center justify-center">
              <span className="text-white font-bold text-sm">SIH</span>
            </div>
            <div className="text-left">
              <div className="text-lg font-semibold text-foreground">Smart India Hackathon 2025</div>
              <div className="text-sm text-muted-foreground">Prototype developed for POWERGRID Corporation</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default PrototypeDemo;