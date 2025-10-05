import { Target, TrendingUp, Package, Clock, DollarSign, BarChart3 } from "lucide-react";

const KPIMetrics = () => {
  const kpis = [
    {
      icon: Clock,
      title: "Reduction in Project Delays",
      description: "Minimize delays caused by material shortages",
      target: "40-60%",
      color: "text-success",
      bgColor: "bg-success/10"
    },
    {
      icon: Package,
      title: "Warehouse Utilization Improvement", 
      description: "Optimize storage space and inventory turnover",
      target: "25-35%",
      color: "text-primary",
      bgColor: "bg-primary/10"
    },
    {
      icon: TrendingUp,
      title: "Reduction in Overstocking",
      description: "Eliminate excess inventory and holding costs",
      target: "30-45%", 
      color: "text-accent",
      bgColor: "bg-accent/10"
    },
    {
      icon: DollarSign,
      title: "Cost Savings per Project",
      description: "Reduced procurement costs and waste elimination",
      target: "₹5-15 Cr",
      color: "text-secondary",
      bgColor: "bg-secondary/10"
    }
  ];

  const impactAreas = [
    {
      title: "Procurement Efficiency",
      metrics: ["Just-in-time delivery", "Reduced emergency orders", "Better supplier negotiations"],
      icon: Target
    },
    {
      title: "Project Timeline",
      metrics: ["Faster project completion", "Predictable milestones", "Reduced idle time"],
      icon: Clock
    },
    {
      title: "Financial Performance", 
      metrics: ["Lower holding costs", "Reduced waste", "Optimized cash flow"],
      icon: BarChart3
    }
  ];

  return (
    <section className="py-24 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-flex items-center space-x-2 bg-success/10 backdrop-blur-sm border border-success/20 rounded-full px-4 py-2 mb-6">
            <Target className="w-4 h-4 text-success" />
            <span className="text-sm text-success">Performance Tracking</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-bold text-gradient mb-6">
            Measuring Impact
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Track measurable improvements in POWERGRID operations through key performance indicators
          </p>
        </div>

        {/* Main KPIs */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          {kpis.map((kpi, index) => (
            <div key={index} className="glass-card p-6 text-center hover-lift">
              <div className={`w-16 h-16 ${kpi.bgColor} rounded-2xl flex items-center justify-center mx-auto mb-6`}>
                <kpi.icon className={`w-8 h-8 ${kpi.color}`} />
              </div>
              <div className={`text-3xl font-bold ${kpi.color} mb-2`}>
                {kpi.target}
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-3">
                {kpi.title}
              </h3>
              <p className="text-sm text-muted-foreground">
                {kpi.description}
              </p>
            </div>
          ))}
        </div>

        {/* Impact Areas */}
        <div className="glass-card p-8 mb-12 animate-slide-up">
          <h3 className="text-2xl font-semibold text-foreground mb-8 text-center">
            Impact Analysis Framework
          </h3>
          <div className="grid md:grid-cols-3 gap-8">
            {impactAreas.map((area, index) => (
              <div key={index} className="text-center">
                <div className="w-12 h-12 bg-gradient-primary rounded-xl flex items-center justify-center mx-auto mb-4">
                  <area.icon className="w-6 h-6 text-white" />
                </div>
                <h4 className="text-lg font-semibold text-foreground mb-4">{area.title}</h4>
                <ul className="space-y-2">
                  {area.metrics.map((metric, idx) => (
                    <li key={idx} className="text-sm text-muted-foreground flex items-center justify-center">
                      <div className="w-2 h-2 bg-accent rounded-full mr-3" />
                      {metric}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        {/* Transition Text */}
        <div className="text-center glass-card p-6">
          <p className="text-lg text-muted-foreground">
            From these outputs, POWERGRID teams can track measurable impact across all infrastructure projects…
          </p>
        </div>
      </div>
    </section>
  );
};

export default KPIMetrics;