import { ArrowRight, Play, TrendingUp, Zap, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import heroBg from "@/assets/hero-bg.jpg";

const Hero = () => {
  const stats = [
    { value: "94%", label: "Accuracy in Demand Prediction" },
    { value: "60%", label: "Reduction in Inventory Costs" },
    { value: "2.5x", label: "Faster Procurement Cycles" },
    { value: "SIH 2025", label: "Prototype Development" },
  ];

  const features = [
    { icon: TrendingUp, text: "AI-Powered Forecasting" },
    { icon: Zap, text: "Real-time Analytics" },
    { icon: Shield, text: "Enterprise Security" },
  ];

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background */}
      <div 
        className="absolute inset-0 bg-cover bg-center opacity-20"
        style={{ backgroundImage: `url(${heroBg})` }}
      />
      <div className="absolute inset-0 hero-gradient opacity-60" />
      
      {/* Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <div className="animate-fade-in">
          {/* Badge */}
          <div className="inline-flex items-center space-x-2 bg-white/10 backdrop-blur-sm border border-white/20 rounded-full px-4 py-2 mb-8">
            <Zap className="w-4 h-4 text-accent" />
            <span className="text-sm text-white/90">Powered by Advanced Machine Learning</span>
          </div>

          {/* Headline */}
          <h1 className="text-4xl sm:text-6xl lg:text-7xl font-bold text-white mb-6 leading-tight">
            Transform Your{" "}
            <span className="text-accent-gradient animate-gradient-shift">
              Supply Chain
            </span>{" "}
            with AI
          </h1>

          <div className="mb-8">
            <p className="text-lg sm:text-xl text-accent font-medium mb-4">
              AI-powered demand forecasting for POWERGRID projects — Prototype built under SIH 2025.
            </p>
            
            <p className="text-xl sm:text-2xl text-white/80 mb-4 max-w-4xl mx-auto leading-relaxed">
              Predict materials demand with 94% accuracy using advanced machine learning. 
              Optimize procurement, reduce inventory costs, and eliminate stockouts.
            </p>
            
            <p className="text-lg text-white/70 max-w-3xl mx-auto">
              Here's why this matters for POWERGRID…
            </p>
          </div>

          {/* Feature Pills */}
          <div className="flex flex-wrap justify-center gap-4 mb-12">
            {features.map((feature, index) => (
              <div
                key={index}
                className="flex items-center space-x-2 bg-white/10 backdrop-blur-sm border border-white/20 rounded-full px-4 py-2 hover-lift"
              >
                <feature.icon className="w-4 h-4 text-accent" />
                <span className="text-white/90 text-sm">{feature.text}</span>
              </div>
            ))}
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16">
            <Button variant="hero" size="xl" className="group">
              Explore Prototype Dashboard
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Button>
            <Button variant="glass" size="xl" className="group">
              <Play className="w-5 h-5 mr-2" />
              View Demo
            </Button>
          </div>
          
          {/* SIH 2025 Badge */}
          <div className="flex justify-center mb-16">
            <div className="inline-flex items-center space-x-3 bg-white/10 backdrop-blur-sm border border-white/20 rounded-full px-6 py-3">
              <div className="w-8 h-8 bg-accent rounded-full flex items-center justify-center">
                <span className="text-white font-bold text-xs">SIH</span>
              </div>
              <span className="text-white/90 text-sm font-medium">Built under SIH 2025 — Open to POWERGRID feedback</span>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 max-w-4xl mx-auto">
            {stats.map((stat, index) => (
              <div key={index} className="glass-card p-6 hover-lift">
                <div className="text-3xl sm:text-4xl font-bold text-gradient mb-2">
                  {stat.value}
                </div>
                <div className="text-white/70 text-sm">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Floating Elements */}
      <div className="absolute top-20 left-10 w-20 h-20 bg-primary/20 rounded-full blur-xl animate-float" />
      <div className="absolute bottom-20 right-10 w-32 h-32 bg-accent/20 rounded-full blur-xl animate-float" style={{ animationDelay: "2s" }} />
      <div className="absolute top-1/2 left-1/4 w-16 h-16 bg-secondary/20 rounded-full blur-xl animate-float" style={{ animationDelay: "4s" }} />
    </section>
  );
};

export default Hero;