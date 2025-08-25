#include <driver/i2s.h>

// I2S Microphone pins
#define I2S_MIC_SERIAL_DATA 32
#define I2S_MIC_BCLK 14
#define I2S_MIC_LRCL 15

// I2S Configuration
#define SAMPLE_RATE 16000
#define BUFFER_SIZE 1024

void setup() {
  Serial.begin(921600);
  
  // I2S Microphone setup
  i2s_config_t i2s_mic_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = BUFFER_SIZE,
    .use_apll = false
  };

  // Corrected pin configuration with proper field order
  i2s_pin_config_t mic_pins = {
    .bck_io_num = I2S_MIC_BCLK,
    .ws_io_num = I2S_MIC_LRCL,
    .data_out_num = I2S_PIN_NO_CHANGE,  // Must come before data_in_num
    .data_in_num = I2S_MIC_SERIAL_DATA
  };

  i2s_driver_install(I2S_NUM_0, &i2s_mic_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &mic_pins);
}

void loop() {
  int16_t samples[BUFFER_SIZE];
  size_t bytes_read = 0;
  
  i2s_read(I2S_NUM_0, &samples, BUFFER_SIZE * sizeof(int16_t), &bytes_read, portMAX_DELAY);
  Serial.write((uint8_t*)samples, bytes_read);
}