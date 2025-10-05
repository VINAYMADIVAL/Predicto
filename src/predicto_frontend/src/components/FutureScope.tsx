import { Lightbulb, ArrowUpRight, Sparkles } from "lucide-react";
import { Card } from "@/components/ui/card";

const FutureScope = () => {
  const futureFeatures = [
    {
      title: "Advanced ML Models",
      description: "Integration of ensemble methods, deep learning (LSTM, Transformers) for improved accuracy",
      icon: Sparkles,
    },
    {
      title: "Real-Time Integration",
      description: "Live data feeds from ERP systems, weather APIs, and market indicators for dynamic forecasting",
      icon: ArrowUpRight,
    },
    {
      title: "Multi-Project Optimization",
      description: "Cross-project resource allocation and portfolio-level demand planning",
      icon: Lightbulb,
    },
    {
      title: "Predictive Maintenance",
      description: "Equipment failure prediction to factor into material demand forecasting",
      icon: Sparkles,
    },
    {
      title: "Supplier Network Integration",
      description: "Connect with supplier systems for automated procurement and just-in-time delivery",
      icon: ArrowUpRight,
    },
    {
      title: "Mobile Application",
      description: "Field team access to forecasts and procurement alerts on mobile devices",
      icon: Lightbulb,
    },
  ];

  return (
    <section id="future-scope" className="py-24 relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-flex items-center space-x-2 bg-accent/10 backdrop-blur-sm border border-accent/20 rounded-full px-4 py-2 mb-6">
            <Lightbulb className="w-4 h-4 text-accent" />
            <span className="text-sm text-accent">Beyond the Prototype</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-bold text-gradient mb-6">
            Future Scope
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Potential enhancements and features for full-scale deployment beyond this SIH 2025 prototype
          </p>
        </div>

        {/* Future Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          {futureFeatures.map((feature, index) => (
            <Card
              key={index}
              className="glass-card p-6 hover-lift animate-slide-up"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="w-12 h-12 bg-gradient-to-r from-accent to-primary rounded-xl flex items-center justify-center mb-4">
                <feature.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-3">
                {feature.title}
              </h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                {feature.description}
              </p>
            </Card>
          ))}
        </div>

        {/* Scalability Note */}
        <div className="glass-card p-8 text-center">
          <h3 className="text-2xl font-bold text-foreground mb-4">
            Scalability Potential
          </h3>
          <p className="text-muted-foreground max-w-3xl mx-auto mb-6">
            This prototype demonstrates core forecasting capabilities. With POWERGRID's historical data 
            and operational integration, the system can scale to handle thousands of concurrent projects, 
            real-time data streams, and enterprise-wide deployment across all regional offices.
          </p>
          <div className="flex flex-wrap justify-center gap-4 text-sm">
            <span className="bg-primary/10 text-primary px-4 py-2 rounded-full">
              Multi-tenant Architecture
            </span>
            <span className="bg-secondary/10 text-secondary px-4 py-2 rounded-full">
              Cloud & On-Premise Options
            </span>
            <span className="bg-accent/10 text-accent px-4 py-2 rounded-full">
              API-First Design
            </span>
          </div>
        </div>
      </div>

      {/* Background Elements */}
      <div className="absolute top-0 left-0 w-72 h-72 bg-accent/5 rounded-full blur-3xl -translate-x-36 -translate-y-36" />
      <div className="absolute bottom-0 right-0 w-72 h-72 bg-primary/5 rounded-full blur-3xl translate-x-36 translate-y-36" />
    </section>
  );
};

export default FutureScope;
