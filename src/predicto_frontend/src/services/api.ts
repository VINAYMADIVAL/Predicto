// ================================================================
// API Service for FastAPI Backend Integration
// ================================================================

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

export interface RegisterData {
  company_name: string;
  email: string;
  gstin: string;
  password: string;
}

export interface LoginData {
  email: string;
  password: string;
}

export interface ForecastInput {
  Budget_Cr: number;
  Tower_Count: number;
  Substations_Count: number;
  Voltage: number;
  Tower_Type: string;
  Circuit: string;
  Line_Length_CKM: number;
  Transformation_Capacity_MVA: number;
  Terrain_Difficulty: string;
  Substation_Type: string;
  Tax_Rate: number;
  Geographic_Region: string;
  Location: string;
  Start_Year: number;
  Completion_Year: number;
  Start_Month: number;
}

export interface ForecastResponse {
  forecast_summary: any;
  quarterly_forecast: any[];
  procurement_plan: any[];
  confidence_intervals: any;
  company: {
    id: number;
    company_name: string;
    gstin: string;
  };
}

class ApiService {
  private getToken(): string | null {
    return localStorage.getItem('access_token');
  }

  private setToken(token: string): void {
    localStorage.setItem('access_token', token);
  }

  private clearToken(): void {
    localStorage.removeItem('access_token');
  }

  async register(data: RegisterData): Promise<{ success: boolean; error?: string }> {
    try {
      const formData = new URLSearchParams();
      formData.append('company_name', data.company_name);
      formData.append('email', data.email);
      formData.append('gstin', data.gstin);
      formData.append('password', data.password);

      const response = await fetch(`${API_BASE_URL}/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData.toString(),
        credentials: 'include',
      });

      if (!response.ok) {
        const error = await response.json();
        return { success: false, error: error.detail || 'Registration failed' };
      }

      // Extract token from Set-Cookie if available
      const cookies = response.headers.get('set-cookie');
      if (cookies) {
        const tokenMatch = cookies.match(/access_token=([^;]+)/);
        if (tokenMatch) {
          this.setToken(tokenMatch[1]);
        }
      }

      return { success: true };
    } catch (error) {
      return { success: false, error: 'Network error. Please try again.' };
    }
  }

  async login(data: LoginData): Promise<{ success: boolean; error?: string }> {
    try {
      const formData = new URLSearchParams();
      formData.append('email', data.email);
      formData.append('password', data.password);

      const response = await fetch(`${API_BASE_URL}/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData.toString(),
        credentials: 'include',
      });

      if (!response.ok) {
        const error = await response.json();
        return { success: false, error: error.detail || 'Login failed' };
      }

      // Extract token from Set-Cookie if available
      const cookies = response.headers.get('set-cookie');
      if (cookies) {
        const tokenMatch = cookies.match(/access_token=([^;]+)/);
        if (tokenMatch) {
          this.setToken(tokenMatch[1]);
        }
      }

      return { success: true };
    } catch (error) {
      return { success: false, error: 'Network error. Please try again.' };
    }
  }

  async logout(): Promise<void> {
    try {
      await fetch(`${API_BASE_URL}/logout`, {
        method: 'POST',
        credentials: 'include',
      });
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      this.clearToken();
    }
  }

  async forecast(data: ForecastInput): Promise<ForecastResponse> {
    const token = this.getToken();
    const response = await fetch(`${API_BASE_URL}/api/forecast`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
      },
      credentials: 'include',
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Forecast failed');
    }

    return response.json();
  }

  async exportForecast(data: ForecastInput): Promise<Blob> {
    const token = this.getToken();
    const response = await fetch(`${API_BASE_URL}/api/export_forecast`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
      },
      credentials: 'include',
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Export failed');
    }

    return response.blob();
  }

  isAuthenticated(): boolean {
    return !!this.getToken();
  }
}

export const apiService = new ApiService();
