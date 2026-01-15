CXX = g++
CXXFLAGS = -O3 -march=native -Iinclude -Isrc -DWHISPER_FFMPEG
LDFLAGS = -lwhisper -lggml -lpthread -ldl -lm -lavcodec -lavformat -lavutil -lswresample

OBJ_DIR = obj
SRC_DIR = src

SRCS = detect-word.cpp $(SRC_DIR)/common.cpp $(SRC_DIR)/common-whisper.cpp $(SRC_DIR)/ffmpeg-transcode.cpp
# This transforms e.g. src/common.cpp -> obj/common.o and detect-word.cpp -> obj/detect-word.o
OBJS = $(addprefix $(OBJ_DIR)/, $(notdir $(SRCS:.cpp=.o)))

all: prepare detect-word

prepare:
	mkdir -p $(OBJ_DIR)

detect-word: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -o detect-word

# Rule for objects in the root directory
$(OBJ_DIR)/detect-word.o: detect-word.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for objects in the src directory
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf detect-word $(OBJ_DIR)
