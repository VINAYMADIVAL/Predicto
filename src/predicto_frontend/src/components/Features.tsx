import { 
  Brain, 
  TrendingUp, 
  Shield, 
  Zap, 
  BarChart3, 
  Target,
  Clock,
  Globe,
  ArrowRight
} from "lucide-react";
import { Button } from "@/components/ui/button";

const Features = () => {
  const mainFeatures = [
    {
      icon: Brain,
      title: "AI-Powered Demand Forecasting",
      description: "Advanced machine learning algorithms analyze historical data, market trends, and external factors to predict materials demand with 94% accuracy.",
      benefits: ["Reduce stockouts by 85%", "Optimize inventory levels", "Predict seasonal variations"],
      color: "from-primary to-primary-glow"
    },
    {
      icon: TrendingUp,
      title: "Intelligent Procurement Planning",
      description: "Automate procurement decisions with real-time insights, supplier performance analytics, and dynamic lead time optimization.",
      benefits: ["Cut procurement costs by 30%", "Optimize supplier selection", "Automate reorder points"],
      color: "from-secondary to-secondary-glow"
    },
    {
      icon: BarChart3,
      title: "Inventory Optimization Engine",
      description: "Balance carrying costs with service levels using sophisticated optimization algorithms that consider demand variability and constraints.",
      benefits: ["Reduce inventory costs by 60%", "Improve cash flow", "Minimize waste"],
      color: "from-accent to-accent-glow"
    }
  ];

  const additionalFeatures = [
    { icon: Shield, title: "Enterprise Security", desc: "SOC 2 compliant with end-to-end encryption" },
    { icon: Zap, title: "Real-time Processing", desc: "Process millions of data points in seconds" },
    { icon: Target, title: "Precision Analytics", desc: "Drill down to SKU-level forecasting accuracy" },
    { icon: Clock, title: "Historical Analysis", desc: "Analyze years of historical data patterns" },
    { icon: Globe, title: "Multi-location Support", desc: "Manage inventory across global operations" },
  ];

  return (
    <section id="features" className="py-24 relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-20 animate-fade-in">
          <div className="inline-flex items-center space-x-2 bg-primary/10 backdrop-blur-sm border border-primary/20 rounded-full px-4 py-2 mb-6">
            <Brain className="w-4 h-4 text-primary" />
            <span className="text-sm text-primary">Powered by Machine Learning</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-bold text-gradient mb-6">
            Transform Your Supply Chain Intelligence
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Harness the power of AI to optimize every aspect of your supply chain, 
            from demand prediction to inventory management.
          </p>
        </div>

        {/* Main Features */}
        <div className="grid lg:grid-cols-3 gap-8 mb-20">
          {mainFeatures.map((feature, index) => (
            <div 
              key={index} 
              className="glass-card p-8 hover-lift group animate-slide-up"
              style={{ animationDelay: `${index * 0.2}s` }}
            >
              <div className={`w-16 h-16 bg-gradient-to-r ${feature.color} rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                <feature.icon className="w-8 h-8 text-white" />
              </div>
              
              <h3 className="text-2xl font-bold text-foreground mb-4">{feature.title}</h3>
              <p className="text-muted-foreground mb-6 leading-relaxed">{feature.description}</p>
              
              <ul className="space-y-2">
                {feature.benefits.map((benefit, idx) => (
                  <li key={idx} className="flex items-center text-sm text-muted-foreground">
                    <div className="w-1.5 h-1.5 bg-accent rounded-full mr-3" />
                    {benefit}
                  </li>
                ))}
              </ul>

              <Button variant="ghost" className="mt-6 text-primary hover:text-accent group">
                Learn More
                <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
              </Button>
            </div>
          ))}
        </div>

        {/* Additional Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
          {additionalFeatures.map((feature, index) => (
            <div 
              key={index}
              className="glass-card p-6 text-center hover-lift animate-scale-in"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="w-12 h-12 bg-gradient-to-r from-primary/20 to-accent/20 rounded-xl flex items-center justify-center mx-auto mb-4">
                <feature.icon className="w-6 h-6 text-primary" />
              </div>
              <h4 className="font-semibold text-foreground mb-2">{feature.title}</h4>
              <p className="text-sm text-muted-foreground">{feature.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Background Decorations */}
      <div className="absolute top-0 left-0 w-96 h-96 bg-primary/5 rounded-full blur-3xl -translate-x-48 -translate-y-48" />
      <div className="absolute bottom-0 right-0 w-96 h-96 bg-accent/5 rounded-full blur-3xl translate-x-48 translate-y-48" />
    </section>
  );
};

export default Features;