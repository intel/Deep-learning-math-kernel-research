CXX       := icc
CXXFLAGS  := -Wall -Werror -Wextra -std=c++11 -fopenmp -fPIC -Iinclude/ -Isrc/
LDFLAGS   := -Llib/
LDLIBS    := -liomp5
TCXX      := icc
TCXXFLAGS := -std=c++11 -fopenmp -pie -fPIE -Iinclude/ -Dlest_FEATURE_AUTO_REGISTER=1
TLDFLAGS   = -Llib -L$(LIB_DIR) -liomp5 -lel -Wl,-rpath=$(LIB_DIR)

ROOT_DIR   = $(shell pwd)
BUILD_DIR := $(ROOT_DIR)/build
DEBUG     ?= 0

ifeq ($(CXX), icc)
CCSP_FLAGS = -qopt-report=5
else
CCSP_FLAGS = -mavx512f
endif

ifeq ($(DEBUG), 1)
CXXFLAGS  += -DDEBUG -g -O0
TCXXFLAGS += -DDEBUG -g -O0
OBJ_DIR   := $(BUILD_DIR)/debug
else
CXXFLAGS  += -O2 -DNDEBUG $(CCSP_FLAGS)
TCXXFLAGS += -O2 -DNDEBUG
OBJ_DIR   := $(BUILD_DIR)/release
endif

SRC_DIR   = src
TEST_DIR  = test
UTEST_DIR = test/unitests
BIN_DIR   = $(OBJ_DIR)/bin
LIB_DIR   = $(OBJ_DIR)/lib

SOURCES   = $(shell ls $(SRC_DIR)/*.cpp)
TESTS     = $(shell ls $(TEST_DIR)/*.cpp)
UTESTS    = $(shell ls $(UTEST_DIR)/*.cpp)
OBJECTS   = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/$(SRC_DIR)/%.o)
TESTOBJS  = $(TESTS:$(TEST_DIR)/%.cpp=$(OBJ_DIR)/$(TEST_DIR)/%.o)
UTESTOBJS = $(UTESTS:$(UTEST_DIR)/%.cpp=$(OBJ_DIR)/$(UTEST_DIR)/%.o)

LIB       = $(LIB_DIR)/libel.so
TEST      = $(BIN_DIR)/elt_conv
UTEST     = $(BIN_DIR)/elt_unitests

.PHONY: lib test debug release distclean print_results
all  : lib test utest print_results
lib  : $(LIB)
test : $(TEST)
utest: $(UTEST)

$(TEST): $(TESTOBJS) $(LIB)
	@mkdir -p $(dir $@)
	$(TCXX) $(TCXXFLAGS) $(TLDFLAGS) -o $@ $(TESTOBJS)

$(UTEST): $(UTESTOBJS) $(LIB)
	@mkdir -p $(dir $@)
	$(TCXX) $(TCXXFLAGS) $(TLDFLAGS) -o $@ $(UTESTOBJS)

$(TESTOBJS): $(OBJ_DIR)/$(TEST_DIR)/%.o : $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(TCXX) $(TCXXFLAGS) -o $@ -c $<

$(UTESTOBJS): $(OBJ_DIR)/$(UTEST_DIR)/%.o : $(UTEST_DIR)/%.cpp
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
	@find $(BUILD_DIR) \( -name "$(shell basename $(LIB))" -o \
                          -name "$(shell basename $(TEST))" -o \
                          -name "$(shell basename $(UTEST))" \) -prune -exec rm -rf {} \;

print_results: lib test utest
	@echo
	@echo Build done:
	@echo "    " $(LIB)
	@echo "    " $(TEST)
	@echo "    " $(UTEST)
	@echo
