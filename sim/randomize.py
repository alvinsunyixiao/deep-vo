import airsim
import datetime
import numpy as np

def randomize_time(client: airsim.VehicleClient):
    now = datetime.datetime.now()
    delta_h = np.random.uniform(-12, 12)
    delta_time = datetime.timedelta(hours=delta_h)
    random_time = now + delta_time
    print(f"Randomized time: {random_time}")
    client.simSetTimeOfDay(True, random_time.strftime("%Y-%m-%d %H:%M:%S"), update_interval_secs=5)

def randomize_weather(client: airsim.VehicleClient,
                      random_exp: float = 0.1):
    has_rain_or_snow = float(np.random.rand() < 0.5)
    has_rain = float(np.random.rand() < 0.5)
    has_snow = 1 - has_rain
    client.simSetWeatherParameter(airsim.WeatherParameter.Rain,
                                  has_rain_or_snow * has_rain * np.random.rand())
    client.simSetWeatherParameter(airsim.WeatherParameter.Roadwetness,
                                  has_rain_or_snow * has_rain * np.random.rand())
    client.simSetWeatherParameter(airsim.WeatherParameter.Snow,
                                  has_rain_or_snow * has_snow * np.random.rand())
    client.simSetWeatherParameter(airsim.WeatherParameter.RoadSnow,
                                  has_rain_or_snow * has_snow * np.random.rand())

    has_leaf = float(np.random.rand() < 0.3)
    client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf,
                                  has_leaf * has_rain * np.random.rand())
    client.simSetWeatherParameter(airsim.WeatherParameter.RoadLeaf,
                                  has_leaf * has_rain * np.random.rand())

    has_fog = float(np.random.rand() < 0.5)
    has_dust = 1 - has_fog
    client.simSetWeatherParameter(airsim.WeatherParameter.Dust,
                                  has_dust * np.random.rand() * .8)
    client.simSetWeatherParameter(airsim.WeatherParameter.Fog,
                                  has_fog * np.random.rand() * .8)

def clear_weather(client: airsim.VehicleClient,
                  random_exp: float = 0.1):
    client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0)
    client.simSetWeatherParameter(airsim.WeatherParameter.Roadwetness, 0)
    client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 0)
    client.simSetWeatherParameter(airsim.WeatherParameter.RoadSnow, 0)
    client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, 0)
    client.simSetWeatherParameter(airsim.WeatherParameter.RoadLeaf, 0)
    client.simSetWeatherParameter(airsim.WeatherParameter.Dust, 0)
    client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0)
