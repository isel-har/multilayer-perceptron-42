CXX    := c++
NAME   := mlp
PYTHON := python3.10

NUMPY_PATH  := /usr/lib/python3/dist-packages/numpy/core/include/
PYTHON_PATH := /usr/include/$(PYTHON)

CXXFLAGS := -O3 -march=corei7 -mavx2 -std=c++17 \
            -Wall -Wextra \
            -Wno-deprecated-copy -Wno-deprecated-declarations


INCLUDES := -I include \
			-isystem lib \
            -isystem lib/eigen \
            -I $(NUMPY_PATH) \
			-I $(PYTHON_PATH)

LIBS := -l$(PYTHON)


SRCS    := $(shell echo src/*.cpp main.cpp)
HEADERS := $(shell echo include/*.hpp include/*.tpp)


OBJS := $(SRCS:.cpp=.o)

LIBDIR := lib

LIB_STAMP := $(LIBDIR)/.libs_done

all: $(LIB_STAMP) $(NAME)


$(LIB_STAMP):
	@echo "Downloading libraries..."
	@mkdir -p $(LIBDIR)

	@wget -nc -P $(LIBDIR) \
	https://raw.githubusercontent.com/d99kris/rapidcsv/refs/heads/master/src/rapidcsv.h \
	https://raw.githubusercontent.com/lava/matplotlib-cpp/refs/heads/master/matplotlibcpp.h \
	https://raw.githubusercontent.com/nlohmann/json/refs/heads/develop/single_include/nlohmann/json.hpp

	@if [ ! -d "$(LIBDIR)/eigen" ]; then \
		echo "Cloning Eigen..."; \
		git clone https://gitlab.com/libeigen/eigen.git $(LIBDIR)/eigen; \
	else \
		echo "Eigen already exists."; \
	fi
	@touch $(LIB_STAMP)



$(NAME): $(OBJS)
	@echo "Linking object files..."
	@$(CXX) $(CXXFLAGS) $(OBJS) $(LIBS) -o $@
	@echo "done"


%.o: %.cpp $(HEADERS)
	@echo "Compiling source files..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@


clean:
	rm -f $(OBJS)

fclean: clean
	rm -f $(NAME)

re: fclean all

.PHONY: all libs clean fclean re
