import { useState, useRef } from 'react';
import { Upload, FileText, AlertCircle, CheckCircle2, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';

interface CSVUploaderProps {
  onDataLoad?: (data: any[]) => void;
  acceptedFileTypes?: string[];
  maxFileSize?: number; // in MB
  className?: string;
}

const CSVUploader = ({ 
  onDataLoad, 
  acceptedFileTypes = ['.csv', '.xlsx', '.xls'],
  maxFileSize = 10,
  className = ""
}: CSVUploaderProps) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [previewData, setPreviewData] = useState<any[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const parseCSV = (text: string): any[] => {
    const lines = text.split('\n').filter(line => line.trim());
    if (lines.length === 0) return [];

    const headers = lines[0].split(',').map(header => header.trim().replace(/"/g, ''));
    const data = lines.slice(1).map(line => {
      const values = line.split(',').map(value => value.trim().replace(/"/g, ''));
      const row: any = {};
      headers.forEach((header, index) => {
        row[header] = values[index] || '';
      });
      return row;
    });

    return data;
  };

  const handleFileUpload = async (file: File) => {
    if (file.size > maxFileSize * 1024 * 1024) {
      toast({
        title: "File too large",
        description: `File size must be less than ${maxFileSize}MB`,
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    setUploadedFile(file);

    try {
      const text = await file.text();
      const data = parseCSV(text);
      
      if (data.length === 0) {
        throw new Error('No valid data found in file');
      }

      setPreviewData(data.slice(0, 5)); // Show first 5 rows as preview
      setUploadStatus('success');
      onDataLoad?.(data);
      
      toast({
        title: "File uploaded successfully",
        description: `Processed ${data.length} rows of data`,
      });
    } catch (error) {
      setUploadStatus('error');
      toast({
        title: "Upload failed",
        description: "Please check your file format and try again",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileSelect = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const clearFile = () => {
    setUploadedFile(null);
    setUploadStatus('idle');
    setPreviewData([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Upload Area */}
      <div
        className={`
          relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200
          ${dragActive 
            ? 'border-primary bg-primary/5' 
            : uploadStatus === 'success' 
              ? 'border-success bg-success/5' 
              : uploadStatus === 'error'
                ? 'border-destructive bg-destructive/5'
                : 'border-border hover:border-primary/50 hover:bg-muted/20'
          }
        `}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={acceptedFileTypes.join(',')}
          onChange={handleFileChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isProcessing}
        />
        
        <div className="flex flex-col items-center space-y-4">
          {isProcessing ? (
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
          ) : uploadStatus === 'success' ? (
            <CheckCircle2 className="w-12 h-12 text-success" />
          ) : uploadStatus === 'error' ? (
            <AlertCircle className="w-12 h-12 text-destructive" />
          ) : (
            <Upload className="w-12 h-12 text-muted-foreground" />
          )}
          
          <div className="space-y-2">
            <h3 className="text-lg font-semibold text-foreground">
              {isProcessing 
                ? 'Processing file...' 
                : uploadedFile 
                  ? uploadedFile.name 
                  : 'Upload your data file'
              }
            </h3>
            <p className="text-sm text-muted-foreground">
              {uploadStatus === 'success' 
                ? `Successfully loaded ${previewData.length > 0 ? '5+' : '0'} rows` 
                : 'Drag and drop your CSV/Excel file here, or click to browse'
              }
            </p>
            <p className="text-xs text-muted-foreground">
              Supports: {acceptedFileTypes.join(', ')} (max {maxFileSize}MB)
            </p>
          </div>

          {!isProcessing && (
            <div className="flex space-x-2">
              <Button variant="outline" size="sm" onClick={handleFileSelect}>
                <FileText className="w-4 h-4 mr-2" />
                Browse Files
              </Button>
              {uploadedFile && (
                <Button variant="ghost" size="sm" onClick={clearFile}>
                  <X className="w-4 h-4 mr-2" />
                  Clear
                </Button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Preview Data */}
      {previewData.length > 0 && (
        <div className="bg-card-glass border border-border/50 rounded-xl p-4">
          <h4 className="text-sm font-medium text-foreground mb-3">Data Preview (first 5 rows)</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/50">
                  {Object.keys(previewData[0] || {}).map((header, index) => (
                    <th key={index} className="text-left py-2 px-3 text-muted-foreground font-medium">
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {previewData.map((row, rowIndex) => (
                  <tr key={rowIndex} className="border-b border-border/20">
                    {Object.values(row).map((value: any, colIndex) => (
                      <td key={colIndex} className="py-2 px-3 text-foreground">
                        {String(value).substring(0, 50)}
                        {String(value).length > 50 && '...'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default CSVUploader;