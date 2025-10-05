import { useState } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  Download, 
  Calendar,
  Filter,
  RefreshCw
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import DemandForecastChart from '@/components/charts/DemandForecastChart';
import InventoryChart from '@/components/charts/InventoryChart';
import CSVUploader from '@/components/data/CSVUploader';

const AnalyticsDashboard = () => {
  const [activeView, setActiveView] = useState<'charts' | 'upload'>('charts');
  const [chartType, setChartType] = useState<'line' | 'bar' | 'area' | 'pie'>('line');
  const [uploadedData, setUploadedData] = useState<any[]>([]);

  const demandData = [
    { month: "Jan", predicted: 85, actual: 82 },
    { month: "Feb", predicted: 92, actual: 89 },
    { month: "Mar", predicted: 78, actual: 81 },
    { month: "Apr", predicted: 95, actual: 93 },
    { month: "May", predicted: 87, actual: 85 },
    { month: "Jun", predicted: 102, actual: 98 },
    { month: "Jul", predicted: 94, actual: 91 },
    { month: "Aug", predicted: 88, actual: 92 },
  ];

  const handleDataUpload = (data: any[]) => {
    setUploadedData(data);
    setActiveView('charts');
  };

  const exportData = () => {
    const dataStr = JSON.stringify(demandData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'demand-forecast-data.json';
    link.click();
  };

  return (
    <section className="py-24 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-flex items-center space-x-2 bg-accent/10 backdrop-blur-sm border border-accent/20 rounded-full px-4 py-2 mb-6">
            <BarChart3 className="w-4 h-4 text-accent" />
            <span className="text-sm text-accent">Advanced Analytics</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-bold text-gradient mb-6">
            Comprehensive Data Analytics
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Upload your data, visualize insights, and make informed decisions with our advanced 
            analytics platform powered by machine learning algorithms.
          </p>
        </div>

        {/* Controls */}
        <div className="glass-card p-6 mb-8 animate-slide-up">
          <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
            <div className="flex items-center space-x-4">
              <Button
                variant={activeView === 'charts' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setActiveView('charts')}
                className="gap-2"
              >
                <BarChart3 className="w-4 h-4" />
                Charts
              </Button>
              <Button
                variant={activeView === 'upload' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setActiveView('upload')}
                className="gap-2"
              >
                <Download className="w-4 h-4" />
                Upload Data
              </Button>
            </div>
            
            <div className="flex items-center space-x-2">
              <Button variant="ghost" size="sm" className="gap-2">
                <Calendar className="w-4 h-4" />
                Last 6 Months
              </Button>
              <Button variant="ghost" size="sm" className="gap-2">
                <Filter className="w-4 h-4" />
                Filter
              </Button>
              <Button variant="ghost" size="sm" onClick={exportData} className="gap-2">
                <Download className="w-4 h-4" />
                Export
              </Button>
              <Button variant="ghost" size="sm" className="gap-2">
                <RefreshCw className="w-4 h-4" />
                Refresh
              </Button>
            </div>
          </div>
        </div>

        {activeView === 'upload' ? (
          <div className="glass-card p-8 animate-fade-in">
            <CSVUploader 
              onDataLoad={handleDataUpload}
              className="max-w-4xl mx-auto"
            />
          </div>
        ) : (
          <div className="space-y-8 animate-fade-in">
            {/* Chart Type Selector */}
            <div className="glass-card p-6">
              <Tabs value={chartType} onValueChange={(value) => setChartType(value as any)}>
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="line">Line Chart</TabsTrigger>
                  <TabsTrigger value="bar">Bar Chart</TabsTrigger>
                  <TabsTrigger value="area">Area Chart</TabsTrigger>
                  <TabsTrigger value="pie">Pie Chart</TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            {/* Charts Grid */}
            <div className="grid lg:grid-cols-2 gap-8">
              {/* Demand Forecasting Chart */}
              <div className="glass-card p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-semibold text-foreground">
                    Demand Forecasting Analysis
                  </h3>
                  <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                    <TrendingUp className="w-4 h-4 text-success" />
                    <span>94.2% Accuracy</span>
                  </div>
                </div>
                <DemandForecastChart 
                  data={uploadedData.length > 0 ? uploadedData.slice(0, 8) : demandData}
                  type={chartType === 'line' || chartType === 'area' ? 'line' : 'bar'}
                />
              </div>

              {/* Inventory Analysis Chart */}
              <div className="glass-card p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-semibold text-foreground">
                    Inventory Analysis
                  </h3>
                  <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                    <BarChart3 className="w-4 h-4 text-primary" />
                    <span>Real-time Data</span>
                  </div>
                </div>
                <InventoryChart 
                  type={chartType === 'pie' ? 'pie' : 'area'}
                />
              </div>
            </div>

            {/* Additional Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="glass-card p-6 text-center">
                <div className="text-3xl font-bold text-primary mb-2">2.4M</div>
                <div className="text-sm text-muted-foreground">Total Cost Savings</div>
                <div className="text-xs text-success mt-1">↑ 15% vs last quarter</div>
              </div>
              <div className="glass-card p-6 text-center">
                <div className="text-3xl font-bold text-accent mb-2">8.5x</div>
                <div className="text-sm text-muted-foreground">Inventory Turnover</div>
                <div className="text-xs text-success mt-1">↑ 12% improvement</div>
              </div>
              <div className="glass-card p-6 text-center">
                <div className="text-3xl font-bold text-warning mb-2">2.1%</div>
                <div className="text-sm text-muted-foreground">Stockout Risk</div>
                <div className="text-xs text-success mt-1">↓ 1.8% reduced</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default AnalyticsDashboard;