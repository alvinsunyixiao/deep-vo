import airsim
import datetime
import threading
import numpy as np

def randomize_time(client: airsim.VehicleClient,
                   client_lock: threading.Lock = threading.Lock()):
    now = datetime.datetime.now()
    delta_h = np.random.uniform(-12, 12)
    delta_time = datetime.timedelta(hours=delta_h)
    random_time = now + delta_time
    print(f"Randomized time: {random_time}")
    with client_lock:
        client.simSetTimeOfDay(True, random_time.strftime("%Y-%m-%d %H:%M:%S"), update_interval_secs=5)

def randomize_weather(client: airsim.VehicleClient,
                      random_exp: float = 0.1,
                      client_lock: threading.Lock = threading.Lock()):
    with client_lock:
        client.simSetWeatherParameter(airsim.WeatherParameter.Rain,
                                      min(np.random.exponential(random_exp), 1))
        client.simSetWeatherParameter(airsim.WeatherParameter.Roadwetness,
                                      min(np.random.exponential(random_exp), 1))
        client.simSetWeatherParameter(airsim.WeatherParameter.Snow,
                                      min(np.random.exponential(random_exp), 1))
        client.simSetWeatherParameter(airsim.WeatherParameter.RoadSnow,
                                      min(np.random.exponential(random_exp), 1))
        client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf,
                                      min(np.random.exponential(random_exp), 1))
        client.simSetWeatherParameter(airsim.WeatherParameter.RoadLeaf,
                                      min(np.random.exponential(random_exp), 1))
        client.simSetWeatherParameter(airsim.WeatherParameter.Dust,
                                      min(np.random.exponential(random_exp), 1))
        client.simSetWeatherParameter(airsim.WeatherParameter.Fog,
                                      min(np.random.exponential(random_exp), 1))
