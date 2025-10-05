import { useState } from "react";
import { 
  TrendingUp, 
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  BarChart3,
  Calendar,
  Package,
  DollarSign
} from "lucide-react";
import { Button } from "@/components/ui/button";
import DemandForecastChart from "@/components/charts/DemandForecastChart";
import InventoryChart from "@/components/charts/InventoryChart";

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState("overview");

  const kpis = [
    {
      title: "Forecast Accuracy",
      value: "94.2%",
      change: "+2.3%",
      trend: "up",
      icon: TrendingUp,
      color: "text-success"
    },
    {
      title: "Inventory Turnover",
      value: "8.5x",
      change: "+15%",
      trend: "up", 
      icon: Package,
      color: "text-primary"
    },
    {
      title: "Cost Savings",
      value: "$2.4M",
      change: "+$340K",
      trend: "up",
      icon: DollarSign,
      color: "text-accent"
    },
    {
      title: "Stockout Risk",
      value: "2.1%",
      change: "-1.8%",
      trend: "down",
      icon: AlertTriangle,
      color: "text-warning"
    }
  ];

  const demandData = [
    { month: "Jan", predicted: 85, actual: 82 },
    { month: "Feb", predicted: 92, actual: 89 },
    { month: "Mar", predicted: 78, actual: 81 },
    { month: "Apr", predicted: 95, actual: 93 },
    { month: "May", predicted: 87, actual: 85 },
    { month: "Jun", predicted: 102, actual: 98 }
  ];

  const alerts = [
    { type: "warning", message: "Steel inventory below reorder point", time: "2h ago" },
    { type: "success", message: "Q2 forecast accuracy exceeded target", time: "4h ago" },
    { type: "info", message: "New supplier data integrated", time: "6h ago" }
  ];

  const tabs = [
    { id: "overview", label: "Overview", icon: BarChart3 },
    { id: "forecasting", label: "Forecasting", icon: TrendingUp },
    { id: "inventory", label: "Inventory", icon: Package },
    { id: "alerts", label: "Alerts", icon: AlertTriangle }
  ];

  return (
    <section id="dashboard" className="py-24 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-flex items-center space-x-2 bg-accent/10 backdrop-blur-sm border border-accent/20 rounded-full px-4 py-2 mb-6">
            <BarChart3 className="w-4 h-4 text-accent" />
            <span className="text-sm text-accent">Real-time Analytics</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-bold text-gradient mb-6">
            Intelligent Dashboard Experience
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Get actionable insights with our intuitive dashboard that transforms complex data 
            into clear, actionable intelligence for your supply chain operations.
          </p>
        </div>

        {/* Dashboard Container */}
        <div className="glass-card p-8 animate-slide-up">
          {/* Dashboard Header */}
          <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center mb-8">
            <div>
              <h3 className="text-2xl font-bold text-foreground mb-2">Supply Chain Command Center</h3>
              <p className="text-muted-foreground flex items-center">
                <Calendar className="w-4 h-4 mr-2" />
                Last updated: 2 minutes ago
              </p>
            </div>
            <Button variant="cta" size="sm">
              Export Report
            </Button>
          </div>

          {/* Tabs */}
          <div className="flex space-x-1 bg-muted/30 rounded-lg p-1 mb-8 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-all duration-200 text-sm font-medium whitespace-nowrap ${
                  activeTab === tab.id
                    ? "bg-primary text-primary-foreground shadow-md"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                }`}
              >
                <tab.icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>

          {/* KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {kpis.map((kpi, index) => (
              <div key={index} className="bg-card-glass border border-border/50 rounded-xl p-6 hover-lift">
                <div className="flex items-center justify-between mb-3">
                  <kpi.icon className={`w-5 h-5 ${kpi.color}`} />
                  <span className={`text-sm font-medium flex items-center ${
                    kpi.trend === "up" ? "text-success" : "text-warning"
                  }`}>
                    {kpi.trend === "up" ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
                    {kpi.change}
                  </span>
                </div>
                <div className="text-2xl font-bold text-foreground mb-1">{kpi.value}</div>
                <div className="text-sm text-muted-foreground">{kpi.title}</div>
              </div>
            ))}
          </div>

          {/* Charts Section */}
          <div className="grid lg:grid-cols-2 gap-8 mb-8">
            {/* Advanced Demand Forecast Chart */}
            <div className="bg-card-glass border border-border/50 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-lg font-semibold text-foreground">Demand Forecasting Analysis</h4>
                <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                  <TrendingUp className="w-4 h-4 text-success" />
                  <span>94.2% Accuracy</span>
                </div>
              </div>
              <div className="h-[300px]">
                <DemandForecastChart data={demandData} type="line" />
              </div>
            </div>

            {/* Alerts Panel */}
            <div className="bg-card-glass border border-border/50 rounded-xl p-6">
              <h4 className="text-lg font-semibold text-foreground mb-4">Recent Alerts</h4>
              <div className="space-y-4">
                {alerts.map((alert, index) => (
                  <div key={index} className="flex items-start space-x-3 p-3 rounded-lg bg-muted/20 hover:bg-muted/30 transition-colors">
                    <div className={`w-2 h-2 rounded-full mt-2 ${
                      alert.type === "warning" ? "bg-warning" :
                      alert.type === "success" ? "bg-success" : "bg-primary"
                    }`} />
                    <div className="flex-1">
                      <p className="text-sm text-foreground">{alert.message}</p>
                      <p className="text-xs text-muted-foreground mt-1">{alert.time}</p>
                    </div>
                  </div>
                ))}
              </div>
              <Button variant="ghost" size="sm" className="w-full mt-4">
                View All Alerts
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Dashboard;