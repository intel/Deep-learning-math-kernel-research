CXX       := icc
CXXFLAGS  := -Wall -Werror -Wextra -std=c++11 -fopenmp -fPIC -Iinclude/ -Isrc/
LDFLAGS   := -Llib/
LDLIBS    := -liomp5
TCXX      := icc
TCXXFLAGS := -std=c++11 -fopenmp -pie -fPIE -Iinclude/
TLDFLAGS   = -Llib -L$(LIB_DIR) -liomp5 -lel -Wl,-rpath=$(LIB_DIR)

BUILD_DIR := build
DEBUG     ?= 0
ifeq ($(DEBUG), 1)
CXXFLAGS  += -DDEBUG -g -O0
TCXXFLAGS += -DDEBUG -g -O0
OBJ_DIR   := $(BUILD_DIR)/debug
else
CXXFLAGS  += -O2 -DNDEBUG -qopt-report=5
TCXXFLAGS += -O2 -DNDEBUG
OBJ_DIR   := $(BUILD_DIR)/release
endif

SRC_DIR   = src
TEST_DIR  = test
BIN_DIR   = $(OBJ_DIR)/bin
LIB_DIR   = $(OBJ_DIR)/lib

SOURCES   = $(shell ls $(SRC_DIR)/*.cpp)
TESTS     = $(shell ls $(TEST_DIR)/*.cpp)
OBJECTS   = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/$(SRC_DIR)/%.o)
TESTOBJS  = $(TESTS:$(TEST_DIR)/%.cpp=$(OBJ_DIR)/$(TEST_DIR)/%.o)

LIB       = $(LIB_DIR)/libel.so
TEST      = $(BIN_DIR)/el_conv

.PHONY: lib test debug release distclean print_results
all  : lib test print_results
lib  : $(LIB)
test : $(TEST)

$(TEST): $(TESTOBJS) $(LIB)
	@mkdir -p $(dir $@)
	$(TCXX) $(TCXXFLAGS) $(TLDFLAGS) -o $@ $(TESTOBJS)

$(TESTOBJS): $(OBJ_DIR)/$(TEST_DIR)/%.o : $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(TCXX) $(TCXXFLAGS) -o $@ -c $<

$(LIB): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) --shared -o $@ $(OBJECTS) $(LDLIBS)

$(OBJECTS): $(OBJ_DIR)/$(SRC_DIR)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	@rm -f $(TEST) $(LIB)
	@find $(OBJ_DIR) \( -name "*.o" -o -name "*.optrpt" \) -exec rm -f {} \;

distclean:
	@find $(BUILD_DIR) \( -name "*.o" -o -name "*.optrpt" \) -exec rm -f {} \;
	@find $(BUILD_DIR) \( -name "$(shell basename $(LIB))" -o -name "$(shell basename $(TEST))" \) -prune -exec rm -rf {} \;

print_results: lib test
	@echo
	@echo Build done:
	@echo "    " $(LIB)
	@echo "    " $(TEST)
	@echo
