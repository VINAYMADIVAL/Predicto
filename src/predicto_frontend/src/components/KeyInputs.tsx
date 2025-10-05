import { Database, MapPin, Calculator, Wrench } from "lucide-react";

const KeyInputs = () => {
  const sampleData = [
    {
      projectId: "PG-2024-001",
      towerType: "220kV Double Circuit",
      substationType: "Grid Substation",
      location: "Maharashtra",
      budget: "₹45 Cr",
      taxSlab: "18% GST",
      steel: "1,250",
      cement: "840"
    },
    {
      projectId: "PG-2024-002", 
      towerType: "400kV Single Circuit",
      substationType: "Switching Station",
      location: "Gujarat",
      budget: "₹32 Cr",
      taxSlab: "12% GST",
      steel: "980",
      cement: "650"
    },
    {
      projectId: "PG-2024-003",
      towerType: "132kV Double Circuit",
      substationType: "Distribution Sub",
      location: "Rajasthan", 
      budget: "₹28 Cr",
      taxSlab: "18% GST",
      steel: "750",
      cement: "520"
    }
  ];

  const inputFactors = [
    { icon: Database, title: "Project Data", description: "Historical project specifications and requirements" },
    { icon: MapPin, title: "Location Factors", description: "Geographic constraints and regional material availability" },
    { icon: Calculator, title: "Financial Data", description: "Budget allocations and tax considerations" },
    { icon: Wrench, title: "Technical Specs", description: "Tower types, substation configurations, and capacity" }
  ];

  return (
    <section className="py-24 bg-muted/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-flex items-center space-x-2 bg-primary/10 backdrop-blur-sm border border-primary/20 rounded-full px-4 py-2 mb-6">
            <Database className="w-4 h-4 text-primary" />
            <span className="text-sm text-primary">Data Inputs</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-bold text-gradient mb-6">
            Key Inputs Considered
          </h2>
        </div>

        {/* Input Factors */}
        <div className="grid md:grid-cols-4 gap-6 mb-12">
          {inputFactors.map((factor, index) => (
            <div key={index} className="glass-card p-6 text-center hover-lift">
              <factor.icon className="w-8 h-8 text-accent mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-foreground mb-2">{factor.title}</h3>
              <p className="text-sm text-muted-foreground">{factor.description}</p>
            </div>
          ))}
        </div>

        {/* Sample Data Table */}
        <div className="glass-card p-8 animate-slide-up">
          <h3 className="text-2xl font-semibold text-foreground mb-6 text-center">
            Sample Input Data Structure
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/50">
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Project ID</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Tower Type</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Substation Type</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Location</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Budget</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Tax Slab</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Steel (tons)</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Cement (tons)</th>
                </tr>
              </thead>
              <tbody>
                {sampleData.map((row, index) => (
                  <tr key={index} className="border-b border-border/30 hover:bg-muted/30 transition-colors">
                    <td className="py-3 px-4 text-primary font-medium">{row.projectId}</td>
                    <td className="py-3 px-4 text-muted-foreground">{row.towerType}</td>
                    <td className="py-3 px-4 text-muted-foreground">{row.substationType}</td>
                    <td className="py-3 px-4 text-muted-foreground">{row.location}</td>
                    <td className="py-3 px-4 text-muted-foreground">{row.budget}</td>
                    <td className="py-3 px-4 text-muted-foreground">{row.taxSlab}</td>
                    <td className="py-3 px-4 text-accent font-semibold">{row.steel}</td>
                    <td className="py-3 px-4 text-accent font-semibold">{row.cement}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-center text-muted-foreground mt-6">
            These factors feed into our forecasting engine…
          </p>
        </div>
      </div>
    </section>
  );
};

export default KeyInputs;