import { Database, MapPin, Brain, TrendingUp, CheckCircle } from "lucide-react";

const HowItWorks = () => {
  const steps = [
    {
      icon: Database,
      title: "Collect Historical Data",
      description: "Gather historical project data, budgets, and material consumption patterns from past POWERGRID projects",
      color: "text-primary"
    },
    {
      icon: MapPin, 
      title: "Add External Factors",
      description: "Incorporate location-specific data, tax implications, geographical constraints, and regional material availability",
      color: "text-secondary"
    },
    {
      icon: Brain,
      title: "Train Forecasting Models",
      description: "Use advanced machine learning algorithms to identify patterns and build predictive models for material demand",
      color: "text-accent"
    },
    {
      icon: TrendingUp,
      title: "Generate Forecasts",
      description: "Output accurate demand forecasts with performance metrics including MAE (Mean Absolute Error) and RMSE (Root Mean Square Error)",
      color: "text-success"
    }
  ];

  const metrics = [
    { label: "MAE", value: "< 5%", description: "Mean Absolute Error" },
    { label: "RMSE", value: "< 8%", description: "Root Mean Square Error" },
    { label: "Accuracy", value: "94.2%", description: "Overall Prediction Accuracy" }
  ];

  return (
    <section className="py-24 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-flex items-center space-x-2 bg-accent/10 backdrop-blur-sm border border-accent/20 rounded-full px-4 py-2 mb-6">
            <Brain className="w-4 h-4 text-accent" />
            <span className="text-sm text-accent">ML Pipeline</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-bold text-gradient mb-6">
            How It Works
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Our AI-powered forecasting engine follows a systematic approach to deliver accurate material demand predictions
          </p>
        </div>

        {/* Process Steps */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
          {steps.map((step, index) => (
            <div key={index} className="relative">
              <div className="glass-card p-8 text-center hover-lift">
                <div className="w-16 h-16 bg-gradient-primary rounded-2xl flex items-center justify-center mx-auto mb-6">
                  <step.icon className="w-8 h-8 text-white" />
                </div>
                <div className="absolute -top-4 -left-4 w-8 h-8 bg-primary rounded-full flex items-center justify-center text-white font-bold text-sm">
                  {index + 1}
                </div>
                <h3 className="text-lg font-semibold text-foreground mb-4">
                  {step.title}
                </h3>
                <p className="text-muted-foreground text-sm">
                  {step.description}
                </p>
              </div>
              {index < steps.length - 1 && (
                <div className="hidden lg:block absolute top-1/2 -right-4 w-8 h-0.5 bg-gradient-to-r from-primary to-accent" />
              )}
            </div>
          ))}
        </div>

        {/* Performance Metrics */}
        <div className="glass-card p-8 text-center animate-slide-up">
          <div className="flex items-center justify-center space-x-2 mb-6">
            <CheckCircle className="w-6 h-6 text-success" />
            <h3 className="text-2xl font-semibold text-foreground">Performance Metrics</h3>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            {metrics.map((metric, index) => (
              <div key={index} className="p-6">
                <div className="text-3xl font-bold text-gradient mb-2">{metric.value}</div>
                <div className="text-lg font-semibold text-foreground mb-1">{metric.label}</div>
                <div className="text-sm text-muted-foreground">{metric.description}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;